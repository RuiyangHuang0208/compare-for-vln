from setuptools import find_packages, setup

package_name = "vln_evaluation"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/evaluation.launch.py", "launch/controller_probe.launch.py"]),
    ],
    install_requires=["numpy", "PyYAML", "setuptools"],
    zip_safe=True,
    maintainer="mifcom2",
    maintainer_email="mifcom2@example.com",
    description="VLN evaluation nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "goal_monitor = vln_evaluation.goal_monitor:main",
            "evaluator = vln_evaluation.evaluator:main",
            "benchmark_runner = vln_evaluation.benchmark_runner:main",
            "controller_probe = vln_evaluation.controller_probe:main",
        ]
    },
)
