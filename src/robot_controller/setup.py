from setuptools import find_packages, setup

package_name = "robot_controller"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/b2w.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mifcom2",
    maintainer_email="mifcom2@example.com",
    description="B2-W ROS-to-Isaac velocity bridge.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["udp_velocity_bridge = robot_controller.udp_velocity_bridge:main"]},
)
