"""Fixtures of small 21cmFAST halo catalogues."""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import py21cmfast as p21c
import pytest
from py21cmfast.wrapper.outputs import HaloCatalog, PerturbedHaloCatalog

HALO_REDSHIFT = 8.0
SMALL_BOX = {
    "BOX_LEN": 20,
    "HII_DIM": 8,
    "DIM": None,
    "HIRES_TO_LOWRES_FACTOR": 2,
    "PERTURB_ALGORITHM": "LINEAR",
}


@pytest.fixture(scope="session")
def halo_inputs() -> p21c.InputParameters:
    """Small inputs without minihalos or reionization feedback."""
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "size-tiny"], random_seed=1
    ).evolve_input_structs(**SMALL_BOX)
    assert not inputs.astro_options.USE_MCGS
    assert not inputs.astro_options.USE_REIONIZATION_PHOTOHEATING_FEEDBACK
    return inputs


@pytest.fixture(scope="session")
def minihalo_inputs() -> p21c.InputParameters:
    """Small inputs with minihalos and relative-velocity fluctuations."""
    inputs = p21c.InputParameters.from_template(
        ["minihalos-discrete", "size-tiny"], random_seed=3
    ).evolve_input_structs(**SMALL_BOX)
    assert inputs.astro_options.USE_MCGS
    assert inputs.matter_options.V_CB_MODEL == "FLUCTS"
    return inputs


@pytest.fixture(scope="session")
def halo_run(halo_inputs) -> SimpleNamespace:
    """Initial conditions and halo catalogues of a tiny run (a few seconds).

    Without minihalos or reionization feedback, the ``PerturbedHaloCatalog`` has the
    halo properties of the galaxy model alone, like the ``HaloCatalog``.
    """
    initial_conditions = p21c.compute_initial_conditions(inputs=halo_inputs)
    halo_catalog = p21c.determine_halo_catalog(
        redshift=HALO_REDSHIFT,
        initial_conditions=initial_conditions,
        inputs=halo_inputs,
    )
    perturbed = p21c.perturb_halo_catalog(
        initial_conditions=initial_conditions,
        inputs=halo_inputs,
        halo_catalog=halo_catalog,
    )
    return SimpleNamespace(
        inputs=halo_inputs,
        initial_conditions=initial_conditions,
        halo_catalog=halo_catalog,
        perturbed=perturbed,
    )


@pytest.fixture
def make_perturbed() -> Callable[..., PerturbedHaloCatalog]:
    """Return a function that makes a PerturbedHaloCatalog with given properties.

    The padding beyond the real halos is NaN, so any use of it shows.
    """

    def make(
        inputs: p21c.InputParameters, n_padding: int = 3, **properties: np.ndarray
    ) -> PerturbedHaloCatalog:
        n_halos = len(next(iter(properties.values())))
        catalog = PerturbedHaloCatalog.new(
            redshift=HALO_REDSHIFT, buffer_size=n_halos + n_padding, inputs=inputs
        )
        properties.setdefault("halo_coords", np.zeros((n_halos, 3)))
        properties.setdefault("halo_masses", np.full(n_halos, 1e10))
        for name, value in properties.items():
            value = np.asarray(value, dtype=np.float32)
            padded = np.full(
                (n_halos + n_padding, *value.shape[1:]), np.nan, dtype=np.float32
            )
            padded[:n_halos] = value
            catalog.set(name, padded)
        catalog.n_halos = n_halos
        return catalog

    return make


@pytest.fixture
def make_halo_catalog() -> Callable[[p21c.InputParameters], HaloCatalog]:
    """Return a function that makes a HaloCatalog of 200 random halos."""

    def make(inputs: p21c.InputParameters) -> HaloCatalog:
        rng = np.random.default_rng(0)
        n_halos = 200
        catalog = HaloCatalog.new(
            inputs=inputs, redshift=HALO_REDSHIFT, buffer_size=n_halos
        )
        catalog.set("halo_masses", (10 ** rng.uniform(6, 11, n_halos)).astype("f4"))
        for name in ("star_rng", "sfr_rng", "xray_rng"):
            catalog.set(name, rng.normal(size=n_halos).astype("f4"))
        box_len = inputs.simulation_options.BOX_LEN
        catalog.set("halo_coords", rng.uniform(0, box_len, (n_halos, 3)).astype("f4"))
        catalog.n_halos = n_halos
        return catalog

    return make
