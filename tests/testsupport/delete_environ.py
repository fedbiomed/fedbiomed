"""
this helper delete the global environ variable if it exists

this may be necessary since the researcher environ and the node environ
have the same name (environ in the global namespace)

beware that nosetests run the tests in the same python process,
so a test may benefit from a previous environment !
"""

import sys


def delete_environ():
    """
    the delete_environ() function must be called by the
    setUpClass() method. eg:

    # only once, before starting the tests of this class
    @classmethod
    def setUpClass(cls):
        from testsupport.delete_environ import delete_environ
        delete_environ()
        pass

    """

    print("Deleting environ")
    for m in ['fedbiomed.common.environ',
              'fedbiomed.node.environ',
              'fedbiomed.researcher.environ',
              'testsupport.mock_common_environ'
              ]:
        if m in sys.modules:
            print("== unloading", m)
            del sys.modules[m]

        # we should also delete the global variable
        if 'environ' in globals():
            print("== deleting environ")
            del globals()['environ']
