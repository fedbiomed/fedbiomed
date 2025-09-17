#
# pytest and cobertura results do not show real test coverage
# but only coverage figures of imported files
#
# by including here all .py files of fedbiomed, we force pytest
# to do the right test coverage calculations
#
# TODO: this file may be automatically crafted on ci plateform
# TODO: this file should only contain files not included in proper test_*.py
#       (this is done by hand right now)
#
# find fedbiomed -name '*.py'  | sort | sed -e 's:.py$::' | sed -e 's:/:.:g'
#

# import fedbiomed.common.dataset._medical_datasets
# import fedbiomed.common.dataset._legacy_tabular_dataset
