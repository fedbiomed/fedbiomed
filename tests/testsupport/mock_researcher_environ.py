import sys
import testsupport.fake_researcher_environ

sys.modules['fedbiomed.researcher.environ'] = testsupport.fake_researcher_environ
