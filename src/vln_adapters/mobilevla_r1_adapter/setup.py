from setuptools import find_packages, setup

package_name = "mobilevla_r1_adapter"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            ["config/mobilevla_r1.yaml", "config/official_archive_manifest.json"],
        ),
        (f"share/{package_name}/launch", ["launch/mobilevla_r1.launch.py"]),
    ],
    install_requires=["numpy", "Pillow", "requests", "setuptools"],
    zip_safe=True,
    maintainer="mifcom2",
    maintainer_email="mifcom2@example.com",
    description="Safe RGB-D MobileVLA-R1 NavigationCommand VELOCITY adapter.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mobilevla_r1_adapter = mobilevla_r1_adapter.mobilevla_r1_node:main",
            "mobilevla_r1_checkpoint = mobilevla_r1_adapter.checkpoint_archive:main",
        ]
    },
)
