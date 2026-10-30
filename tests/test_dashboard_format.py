from nexus.dashboard_format import NA, format_compact, format_delta, format_number, format_pct


def test_format_compact_scales_and_keeps_the_sign():
    assert format_compact(12337953.99) == "12.34M"
    assert format_compact(4100) == "4.10K"
    assert format_compact(950) == "950"
    assert format_compact(-1500000) == "-1.50M"
    assert format_compact(2.5e9) == "2.50B"
    assert format_compact(None) == NA


def test_format_pct_number_and_delta():
    assert format_pct(31.64) == "31.6%"
    assert format_pct(None) == NA
    assert format_number(1234567.891, 2) == "1,234,567.89"
    assert format_number(None) == NA
    assert format_delta(12.34) == "+12.3%"
    assert format_delta(-3.0) == "-3.0%"
    assert format_delta(None) is None