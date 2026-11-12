"""Units against values computed directly with astropy."""

import astropy.units as au
import astropy.constants as ac
import pytest

import pyathena as pa


@pytest.mark.parametrize('muH', [1.4, 1.4271])
def test_lv_units(muH):
    u = pa.Units(kind='LV', muH=muH)
    t = (1.0*au.pc/(au.km/au.s)).to('Myr').value
    m = (muH*1.008*au.u*au.pc**3/au.cm**3).to('Msun').value
    assert u.muH == muH
    assert u.length.to('pc').value == pytest.approx(1.0, rel=1e-12)
    assert u.velocity.to('km s-1').value == pytest.approx(1.0, rel=1e-12)
    assert u.Myr == pytest.approx(t, rel=1e-10)
    assert u.Msun == pytest.approx(m, rel=1e-10)
    assert u.pok == pytest.approx(
        (u.density*u.velocity**2/ac.k_B).cgs.value, rel=1e-10)
