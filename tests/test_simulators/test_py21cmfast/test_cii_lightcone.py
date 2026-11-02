"""Tests of building a [CII] lightcone from coeval boxes."""

from types import SimpleNamespace

import numpy as np
import py21cmfast as p21c
import pytest
from py21cmfast.drivers.lightcone import LightCone

from tuesday.simulators.py21cmfast.postprocessing import cii_lightcone
from tuesday.simulators.py21cmfast.postprocessing.cii_lightcone import gen_lc

NODE_REDSHIFTS = [7.0, 6.5, 6.0]


def make_run() -> tuple[p21c.InputParameters, p21c.RectilinearLightconer, LightCone]:
    """Create small inputs, a lightconer and a lightcone with two plain fields."""
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1, node_redshifts=NODE_REDSHIFTS
    ).evolve_input_structs(BOX_LEN=50, HII_DIM=10, DIM=20, PERTURB_ALGORITHM="LINEAR")
    lightconer = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=6.1,
        max_redshift=6.9,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "density"),
    )
    shape = (10, 10, len(lightconer.lc_distances))
    lightcone = LightCone(
        lightcone_distances=lightconer.lc_distances,
        inputs=inputs,
        lightcones={
            "brightness_temp": np.full(shape, 5.0),
            "density": np.full(shape, 0.1),
        },
    )
    return inputs, lightconer, lightcone


@pytest.fixture
def run(monkeypatch) -> SimpleNamespace:
    """A small run whose cache reading is replaced by uniform [CII] coeval boxes.

    Each box has an intensity equal to its redshift (times ``scale``), so the
    interpolated lightcone lies between the node values. The seed passed by
    ``gen_lc`` is recorded in ``calls``.
    """
    inputs, lightconer, lightcone = make_run()
    calls = {"seed": None, "scale": 1.0}

    def fake_gen_coeval_list(lc, model, directory_path, seed=None):
        calls["seed"] = seed
        return [
            SimpleNamespace(
                redshift=z,
                cosmo_params=inputs.cosmo_params,
                simulation_options=inputs.simulation_options,
                CII_IntensityBox=np.full((10, 10, 10), calls["scale"] * z),
            )
            for z in lc.inputs.node_redshifts
        ]

    monkeypatch.setattr(cii_lightcone, "gen_coeval_list", fake_gen_coeval_list)
    return SimpleNamespace(lightconer=lightconer, lightcone=lightcone, calls=calls)


def test_gen_lc_output(run):
    cii_lc = gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused", seed=3)

    assert isinstance(cii_lc, LightCone)
    intensity = cii_lc.lightcones["CII_IntensityBox"]
    assert intensity.shape == run.lightcone.lightcones["brightness_temp"].shape
    # Interpolated between the node values, and no slice left empty.
    assert np.all(intensity >= min(NODE_REDSHIFTS))
    assert np.all(intensity <= max(NODE_REDSHIFTS))
    assert np.all(cii_lc.lightcones["CII_BrightnessDensity"] > 0)
    assert run.calls["seed"] == 3


def test_gen_lc_keeps_other_fields(run):
    cii_lc = gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")

    np.testing.assert_array_equal(cii_lc.lightcones["brightness_temp"], 5.0)
    np.testing.assert_array_equal(cii_lc.lightcones["density"], 0.1)


def test_gen_lc_does_not_mutate_inputs(run):
    original_tb = run.lightcone.lightcones["brightness_temp"]
    gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")

    assert set(run.lightcone.lightcones) == {"brightness_temp", "density"}
    assert run.lightcone.lightcones["brightness_temp"] is original_tb
    np.testing.assert_array_equal(original_tb, 5.0)
    assert run.lightconer.quantities == ("brightness_temp", "density")


def test_gen_lc_calls_are_independent(run):
    first = gen_lc(run.lightcone, run.lightconer, "Lagache18", "unused")
    first_intensity = first.lightcones["CII_IntensityBox"].copy()

    run.calls["scale"] = 2.0
    second = gen_lc(run.lightcone, run.lightconer, "deLooze14", "unused")

    np.testing.assert_array_equal(first.lightcones["CII_IntensityBox"], first_intensity)
    np.testing.assert_allclose(
        second.lightcones["CII_IntensityBox"], 2 * first_intensity, rtol=1e-12
    )


def test_cii_lightcone_class_removed():
    assert not hasattr(cii_lightcone, "CIILightcone")
