import nexus


def test_package_importable():
    assert nexus.__version__ == "0.1.0"