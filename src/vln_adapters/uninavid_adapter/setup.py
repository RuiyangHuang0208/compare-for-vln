from setuptools import find_packages, setup

package_name = "uninavid_adapter"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/uninavid.yaml"]),
        (f"share/{package_name}/launch", ["launch/uninavid.launch.py"]),
    ],
    install_requires=["numpy", "Pillow", "requests", "setuptools"],
    zip_safe=True,
    maintainer="mifcom2",
    maintainer_email="mifcom2@example.com",
    description="Uni-NaVid NavigationCommand adapter.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["uninavid_adapter = uninavid_adapter.uninavid_node:main"]},
)
