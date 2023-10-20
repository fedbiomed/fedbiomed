import unittest
import asyncio 

from unittest.mock import patch, MagicMock, AsyncMock


from fedbiomed.transport.node_agent import AgentStore
from fedbiomed.transport.server import GrpcServer, _GrpcAsyncServer, ResearcherServicer, NodeAgent
from fedbiomed.common.exceptions import FedbiomedCommunicationError
from fedbiomed.common.message import SearchRequest, SearchReply
from fedbiomed.transport.protocols.researcher_pb2 import TaskRequest, TaskResult, Empty, FeedbackMessage


example_task = SearchRequest(
    researcher_id="r-id",
    tags=["test"],
    command='search'   
)

reply = SearchReply(
    researcher_id='researcher-id',
    node_id='node-id',
    success=True,
    databases=[],
    count=0,
    command='search'
)

class TestResearcherServicer(unittest.IsolatedAsyncioTestCase):

    def setUp(self) -> None:

        self.request = TaskRequest(
            node="node-1",
            protocol_version="x"
        )

        self.context = MagicMock()

        self.agent_store = MagicMock(spec=AgentStore)
        self.on_message = MagicMock()

        self.servicer = ResearcherServicer(
            agent_store=self.agent_store,
            on_message=self.on_message
        )
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()


    async def test_researcher_servicer_01_GetTaskUnary(self):
        
        node_agent = AsyncMock()
        node_agent.set_context = MagicMock()
        node_agent.task_done = MagicMock()
        node_agent.get_task.return_value = example_task

        self.agent_store.retrieve.return_value = node_agent
        async for r in self.servicer.GetTaskUnary(request=self.request, context=self.context):
            self.assertEqual(r.iteration, 1)
            self.assertEqual(r.size, 1)
            

    @patch('fedbiomed.transport.server.Serializer.loads')
    async def test_researcher_servicer_02_ReplyTask(self, load):

        # Creates async iterator
        async def request_iterator():
            for i in [1, 2]:
                yield TaskResult(
                    size=2,
                    iteration=i,
                    bytes_=b'test'
                )

        load.return_value = 'REPLY'
        result = await self.servicer.ReplyTask(request_iterator=request_iterator(), unused_context=self.context)
        self.on_message.assert_called_once()
        self.assertEqual(result, Empty())


    async def test_researcher_servicer_03_Feedback(self):

        request = FeedbackMessage(
            researcher_id='test',
            log=FeedbackMessage.Log(
                node_id='test',
                level='DEBUG',
                msg="Error message"
            )
        )

        result = await self.servicer.Feedback(request=request,  unused_context=self.context)
        self.on_message.assert_called_once()
        self.assertEqual(result, Empty())
    

class TestGrpcAsyncServer(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()


class TestGrpcServer(unittest.TestCase):
    
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()



if __name__ == "__main__":
    unittest.main()