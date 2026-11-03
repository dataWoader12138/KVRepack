import setuptools
from setuptools import setup

# This is to make sure that the package supports editable installs
if __name__ == "__main__":
    setuptools.setup(
        license_files=["LICENSE"],
        include_package_data=True,
        package_data={
            "lmms_eval": [
                "tasks/**/*",
                "tasks/**/_default_template_yaml",
                "models/model_utils/**/*",
            ],
        },
    )
