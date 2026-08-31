from setuptools import find_packages, setup


package_name = "navigation_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/navigation.yaml"]),
        (f"share/{package_name}/launch", ["launch/navigation_bridge.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mifcom2",
    maintainer_email="mifcom2@example.com",
    description="Shared navigation bridge for VLN commands.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["navigation_bridge = navigation_bridge.navigation_bridge_node:main"]},
)
