from setuptools import find_packages, setup

package_name = "dummy_adapter"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/dummy.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mifcom2",
    maintainer_email="mifcom2@example.com",
    description="Dummy VLN adapter.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["dummy_adapter = dummy_adapter.dummy_node:main"]},
)
