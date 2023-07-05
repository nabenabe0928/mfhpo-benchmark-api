import os
import setuptools


min_requirements = []
with open("requirements-minimal.txt", "r") as f:
    for line in f:
        min_requirements.append(line.strip())


pkg_name = "mfhpo-benchmark-api"
author = "nabenabe0928"
main_dir = "benchmark_apis"
pkgs = [main_dir]
dir_names = [fn for fn in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, fn))]
pkgs += [
    os.path.join(main_dir, dir_name) for dir_name in dir_names if all(not dir_name.startswith(s) for s in ["_", "."])
]
extra_requirements = {
    "jahs": ["jahs-bench"],
    "lcbench": ["yahpo-gym"],
    "full": ["jahs-bench", "yahpo-gym"],
}

json_data = [fn for fn in os.listdir("benchmark_apis/hpo") if fn.endswith(".json")]
setuptools.setup(
    name=pkg_name,
    python_requires=">=3.8",
    platforms=["Linux", "Darwin"],
    version="2.0.1",
    author=author,
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url=f"https://github.com/{author}/{pkg_name}",
    packages=pkgs,
    package_data={"": json_data},
    install_requires=min_requirements,
    extras_require=extra_requirements,
    include_package_data=True,
)
