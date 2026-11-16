===========
Lightconers
===========

A simulation gives the fields of the universe at a number of redshifts. A
**lightconer** makes lightcones out of them: it is given the *snapshots* (the boxes of
each redshift) one at a time, from high to low redshift, and returns the lightcone
piece by piece.

The lightconers of ``tuesday`` are in :mod:`tuesday.core.lightcones`. They do not
depend on any simulator. They follow the lightconer protocol of 21cmFAST
(``py21cmfast.lightconers.LightconerProtocol``), so they can be passed directly to
``py21cmfast.run_lightcone``, and they can be used on boxes from anywhere else with
:func:`~tuesday.core.lightcones.build_lightcone`.

.. contents::
   :local:
   :depth: 1


Which lightconers are there?
============================

:class:`~tuesday.core.lightcones.RectilinearLightconer`
    Lightcones that are boxes: two transverse axes and a line-of-sight axis.

:class:`~tuesday.core.lightcones.AngularLightconer`
    Lightcones whose slices are sets of directions on the sky, of shape
    ``(n_pixels, n_slices)``. The directions can be anything: a patch, a ring, or a
    HEALPix map of the full sky (:meth:`~tuesday.core.lightcones.AngularLightconer.healpix`).
    Each pixel can average several directions.

:class:`~tuesday.core.lightcones.BracketingLightconer`
    The base class of both, for writing your own (see below).

All of them make each slice of the lightcone from the two snapshots around it,
interpolated linearly in comoving distance.


With boxes from anywhere
========================

A :class:`~tuesday.core.lightcones.Snapshot` is a redshift and a box per field. A
:class:`~tuesday.core.lightcones.LightconeContext` tells the lightconer about the
boxes. :func:`~tuesday.core.lightcones.build_lightcone` does the rest:

.. code-block:: python

    import numpy as np
    from astropy import units as un
    from astropy.cosmology import Planck18

    from tuesday.core.lightcones import (
        AngularLightconer,
        LightconeContext,
        RectilinearLightconer,
        Snapshot,
        build_lightcone,
    )

    # Some boxes: 32^3 cells of 2 Mpc, at four redshifts.
    rng = np.random.default_rng(0)
    redshifts = (8.0, 7.6, 7.2, 6.8)
    snapshots = [
        Snapshot(z, {"density": rng.normal(size=(32, 32, 32))}) for z in redshifts
    ]
    context = LightconeContext(
        cosmo=Planck18,
        box_shape=(32, 32, 32),
        cell_size=2 * un.Mpc,
        node_redshifts=redshifts,
    )

    between = {
        "min_redshift": 6.9,
        "max_redshift": 7.9,
        "resolution": 2 * un.Mpc,
        "cosmo": Planck18,
        "quantities": ("density",),
    }

    rect = RectilinearLightconer.between_redshifts(**between)
    build_lightcone(rect, snapshots, context)["density"].shape  # (32, 32, 165)

    sky = AngularLightconer.healpix(nside=8, **between)
    build_lightcone(sky, snapshots, context)["density"].shape  # (768, 165)

The snapshots are used one at a time, so ``snapshots`` can be a generator that makes
(or reads) each one when it is asked for. A box can also be given as a function that
returns it, which is only called if the box is used:

.. code-block:: python

    def read(z, field):
        print(f"reading {field} at z={z}")
        return rng.normal(size=(32, 32, 32))

    lazy = (Snapshot(z, {"density": lambda z=z: read(z, "density")}) for z in redshifts)

    # A lightcone between the last two redshifts only reads those two.
    short = rect.evolve(lc_distances=rect.lc_distances[:10])
    build_lightcone(short, lazy, context)["density"].shape  # (32, 32, 10)


With 21cmFAST
=============

The same lightconers are passed to ``py21cmfast.run_lightcone``. Give them the
cosmology of the simulation:

.. code-block:: python

    import py21cmfast as p21c

    inputs = p21c.InputParameters.from_template(
        ["simple", "tiny"], random_seed=1
    ).with_logspaced_redshifts(zmin=7.0, zmax=12.0)
    # The line-of-sight velocity of an angular lightcone needs the 3D velocities.
    inputs = inputs.evolve_input_structs(KEEP_3D_VELOCITIES=True)

    lightconer = AngularLightconer.healpix(
        nside=16,
        subsample_level=1,  # each pixel is the mean over 4 sub-pixels
        min_redshift=7.0,
        max_redshift=10.0,
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    lightcone = p21c.run_lightcone(lightconer=lightconer, inputs=inputs)
    lightcone.lightcones["brightness_temp"].shape  # (3072, 551)

To make lightcones from a run that is already in a cache, without running anything,
see :func:`~tuesday.simulators.py21cmfast.lightcones.construct_lightcone_from_cache`,
and :func:`~tuesday.simulators.py21cmfast.lightcones.make_lightcones_on_disk` for
lightcones that do not fit in memory.

.. note::

    ``tuesday.simulators.py21cmfast.lightconers.FastAngularLightconer`` is now
    :class:`~tuesday.core.lightcones.AngularLightconer`. The old name still works, with
    a deprecation warning.


Writing a lightconer
====================

A lightconer that makes each slice from the two snapshots around it only needs to say
*how one slice is made*. Subclass
:class:`~tuesday.core.lightcones.BracketingLightconer`, and define:

``lightcone_shape(context)``
    The shape of the lightcone of each quantity (line of sight last).

``make_field_slice(near, far, slice_, session, kind)``
    One slice, from the boxes of a field in the snapshots on the near (low-redshift)
    and far side of it. ``slice_`` has the ``index`` and ``distance`` of the slice,
    and ``slice_.interpolate(a, b)`` interpolates two values (or arrays) at the two
    snapshots to the distance of the slice.

``prepare_slice(slice_, session)`` (optional)
    Called once for each slice, before its quantities are made. What it returns is
    available as ``slice_.state`` for every quantity of the slice.

This lightconer makes the lightcone of one line of sight through the box:

.. code-block:: python

    import attrs

    from tuesday.core.lightcones import BracketingLightconer


    @attrs.define(kw_only=True, slots=False)
    class ColumnLightconer(BracketingLightconer):
        """A lightcone of one line of sight through the box: shape (1, n_slices)."""

        x: int = 0
        y: int = 0

        def lightcone_shape(self, context):
            return (1, len(self.lc_distances))

        def prepare_slice(self, slice_, session):
            # The cell along the line of sight, shared by all quantities.
            return slice_.index % session.context.box_shape[2]

        def make_field_slice(self, near, far, slice_, session, kind="mean"):
            k = slice_.state
            return slice_.interpolate(
                near[self.x, self.y, k], far[self.x, self.y, k], kind=kind
            )


    column = ColumnLightconer.between_redshifts(x=3, y=4, **between)
    build_lightcone(column, snapshots, context)["density"].shape  # (1, 165)

Everything else comes from the base class: the description of the lightcone
(``lc_distances`` or ``lc_redshifts``, ``quantities``, ``cosmo``,
:meth:`~tuesday.core.lightcones.BracketingLightconer.between_redshifts`), finding the
slices between each pair of snapshots, and the ``"los_velocity"`` quantity (made from
the fields named in ``los_velocity_fields``, by ``make_los_velocity_slice``).


Caching
-------

Making a slice often needs something expensive that can be used again: an
interpolator onto the slice that is the same for every quantity, a set of
coordinates that is the same for every slice, a box computed from both snapshots.
The session has a :class:`~tuesday.core.lightcones.SessionCache`, with one dictionary
per lifetime:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Dictionary
     - Emptied
     - For what depends on
   * - ``session.cache.slice``
     - after each slice
     - the distance of the slice (but not the quantity)
   * - ``session.cache.interval``
     - after each pair of snapshots
     - the two snapshots (but not the slice)
   * - ``session.cache.run``
     - at the end of the run
     - neither

What ``prepare_slice`` returns (``slice_.state``) is the simplest case of the first.
For the others, put things in the dictionaries yourself:

.. code-block:: python

    @attrs.define(kw_only=True, slots=False)
    class ContrastLightconer(RectilinearLightconer):
        """Adds the quantity "contrast": the density over the mean of the far box."""

        def fields_for(self, quantity):
            # The fields that a quantity is made from.
            if quantity == "contrast":
                return ("density",)
            return super().fields_for(quantity)

        def make_slice(self, quantity, slice_, session):
            if quantity != "contrast":
                return super().make_slice(quantity, slice_, session)

            cache = session.cache.interval
            if "mean" not in cache:
                # Once for all the slices between these two snapshots.
                cache["mean"] = np.mean(slice_.far["density"])
            density = super().make_slice("density", slice_, session)
            return density / cache["mean"] - 1


    contrast = ContrastLightconer.between_redshifts(
        **{**between, "quantities": ("density", "contrast")}
    )
    sorted(build_lightcone(contrast, snapshots, context))  # ['contrast', 'density']

``make_slice`` is the method that makes one quantity on one slice. By default it
calls ``make_field_slice`` with the boxes of the field of that name (or
``make_los_velocity_slice``), so it is also where a quantity that is not simply a
field is defined, with the fields it needs in ``fields_for``.


Other kinds of lightconer
-------------------------

A lightconer does not have to be a ``BracketingLightconer``. What a driver (such as
``py21cmfast.run_lightcone`` or :func:`~tuesday.core.lightcones.build_lightcone`)
needs is:

* ``lightconer.quantities`` and ``lightconer.lc_distances``;
* ``lightconer.required_fields()`` and ``lightconer.evolve(**changes)``;
* ``lightconer.start(context)``, which returns a *session* with a ``shape``, and the
  methods ``ingest(snapshot)`` and ``finish()``, which return
  :class:`~tuesday.core.lightcones.Chunk` objects: ``(quantity, index, values)``,
  meaning ``lightcone[quantity][index] = values``.

A session keeps whatever it needs between snapshots, so a lightconer that needs more
than the two snapshots around a slice (a higher-order interpolation in time, say) is
written by keeping more of them. The documentation of 21cmFAST has a worked example.
