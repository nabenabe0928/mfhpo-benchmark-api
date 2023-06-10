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

setuptools.setup(
    name=pkg_name,
    python_requires=">=3.8",
    platforms=["Linux", "Darwin"],
    version="1.2.0",
    author=author,
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url=f"https://github.com/{author}/{pkg_name}",
    packages=pkgs,
    package_data={"": ["discrete_search_spaces.json"]},
    install_requires=min_requirements,
    extras_require=extra_requirements,
    include_package_data=True,
)
