import numpy as np
import dask
import dask.array as da
import dask.bag as db


def strfunc_from_pdf1(rxs, pdf, values, order, absolute=False):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    if absolute:
        values = abs(values)

    irx_max = rxs.size

    n = pdf.shape[1]
    dpdf = da.from_array(pdf, chunks=(1, n))
    dvalues = da.from_array(values, chunks=(1, n))
    S_order = da.empty(rxs.shape, chunks=1)
    print(f"S_order {S_order.shape}")
    print(f"pdf {pdf.shape}")
    print(f"values {values.shape}")
    for irx in range(irx_max):
        S_order[irx] = da.sum(dpdf[irx] * dvalues[irx] ** order) * np.abs(
            dvalues[irx, 1] - dvalues[irx, 0]
        )

    return S_order.compute()


def strfunc_from_pdf2(rxs, pdf, values, order, absolute=False):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    if absolute:
        values = abs(values)

    irx_max = rxs.size

    n = pdf.shape[1]
    dpdf = da.from_array(pdf, chunks=(1, n))
    dvalues = da.from_array(values, chunks=(1, n))
    S_order = []
    print(f"pdf {pdf.shape}")
    print(f"values {values.shape}")
    for irx in range(irx_max):
        deltainc = abs(dvalues[irx, 1] - dvalues[irx, 0])
        S_order.append(deltainc * da.sum(dpdf[irx] * dvalues[irx] ** order))

    dS_order = da.concatenate(S_order, axis=-1)
    return dS_order.compute()


def strfunc_from_pdf3(rxs, pdf, values, order, absolute=False):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    if absolute:
        values = abs(values)

    irx_max = rxs.size

    def S_order_irx(irx, pdf, values):
        deltainc = abs(values[irx, 1] - values[irx, 0])
        return deltainc * np.sum(pdf[irx] * values[irx] ** order)

    S_order = [
        dask.delayed(S_order_irx)(irx, pdf, values) for irx in range(irx_max)
    ]
    return np.array(
        dask.compute(S_order, scheduler="multiprocessing", num_workers=6)[0]
    )


def strfunc_from_pdf4(rxs, pdf, values, order, absolute=False):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    if absolute:
        values = abs(values)

    irx_max = rxs.size

    def S_order_irx(ipdf, ivalues):
        # print(f"ipdf {ipdf.shape}")
        # print(f"ivalues {ivalues.shape}")
        ipdf = ipdf[0]
        ivalues = ivalues[0]
        deltainc = abs(ivalues[1] - ivalues[0])
        return np.array(deltainc * np.sum(ipdf * ivalues ** order))

    n = pdf.shape[1]
    print(f"pdf {pdf.shape}")
    print(f"values {values.shape}")
    dpdf = da.from_array(pdf, chunks=(1, n))
    dvalues = da.from_array(values, chunks=(1, n))
    # S_order = da.empty(rxs.shape, chunks=1)

    S_order = da.map_blocks(S_order_irx, dpdf, dvalues, dtype=float)
    return S_order.compute()


def strfunc_from_pdf5(rxs, pdf, values, order, absolute=False):
    """Compute structure function of specified order from pdf for increments
    module.

    """
    if absolute:
        values = abs(values)

    irx_max = rxs.size
    print(irx_max)

    def S_order_irx(irx):  # , pdf, values):
        deltainc = abs(values[irx, 1] - values[irx, 0])
        return deltainc * np.sum(pdf[irx] * values[irx] ** order)

    S_order = db.from_sequence(range(irx_max), npartitions=20).map(S_order_irx)
    return np.array(S_order.compute())
