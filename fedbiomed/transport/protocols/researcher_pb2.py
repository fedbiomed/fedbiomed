# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fedbiomed/transport/protocols/researcher.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n.fedbiomed/transport/protocols/researcher.proto\x12\nresearcher\"\x07\n\x05\x45mpty\"+\n\x0fProtocolVersion\x12\x18\n\x10protocol_version\x18\x65 \x01(\t\"\xa9\x05\n\x0f\x46\x65\x65\x64\x62\x61\x63kMessage\x12\x18\n\x10protocol_version\x18\x01 \x01(\t\x12\x1a\n\rresearcher_id\x18\x02 \x01(\tH\x01\x88\x01\x01\x12\x34\n\x06scalar\x18\x03 \x01(\x0b\x32\".researcher.FeedbackMessage.ScalarH\x00\x12.\n\x03log\x18\x04 \x01(\x0b\x32\x1f.researcher.FeedbackMessage.LogH\x00\x1a\xa2\x03\n\x06Scalar\x12\x0f\n\x07node_id\x18\x01 \x01(\t\x12\x0e\n\x06job_id\x18\x02 \x01(\t\x12\r\n\x05train\x18\x03 \x01(\x08\x12\x0c\n\x04test\x18\x04 \x01(\x08\x12\x1d\n\x15test_on_local_updates\x18\x05 \x01(\x08\x12\x1e\n\x16test_on_global_updates\x18\x06 \x01(\x08\x12>\n\x06metric\x18\x07 \x03(\x0b\x32..researcher.FeedbackMessage.Scalar.MetricEntry\x12\x12\n\x05\x65poch\x18\x08 \x01(\x05H\x00\x88\x01\x01\x12\x15\n\rtotal_samples\x18\t \x01(\x05\x12\x15\n\rbatch_samples\x18\n \x01(\x05\x12\x13\n\x0bnum_batches\x18\x0b \x01(\x05\x12 \n\x13num_samples_trained\x18\x0c \x01(\x05H\x01\x88\x01\x01\x12\x11\n\titeration\x18\r \x01(\x05\x1a-\n\x0bMetricEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x02:\x02\x38\x01\x42\x08\n\x06_epochB\x16\n\x14_num_samples_trained\x1a\x32\n\x03Log\x12\x0f\n\x07node_id\x18\x01 \x01(\t\x12\r\n\x05level\x18\x02 \x01(\t\x12\x0b\n\x03msg\x18\x03 \x01(\tB\x0f\n\rfeedback_typeB\x10\n\x0e_researcher_id\"5\n\x0bTaskRequest\x12\x0c\n\x04node\x18\x01 \x01(\t\x12\x18\n\x10protocol_version\x18\x02 \x01(\t\"?\n\x0cTaskResponse\x12\x0c\n\x04size\x18\x01 \x01(\x05\x12\x11\n\titeration\x18\x02 \x01(\x05\x12\x0e\n\x06\x62ytes_\x18\x03 \x01(\x0c\"=\n\nTaskResult\x12\x0c\n\x04size\x18\x01 \x01(\x05\x12\x11\n\titeration\x18\x02 \x01(\x05\x12\x0e\n\x06\x62ytes_\x18\x03 \x01(\x0c\"#\n\x11TaskResponseUnary\x12\x0e\n\x06\x62ytes_\x18\x01 \x01(\x0c\x32\x98\x02\n\x11ResearcherService\x12\x42\n\x07GetTask\x12\x17.researcher.TaskRequest\x1a\x18.researcher.TaskResponse\"\x00(\x01\x30\x01\x12\x45\n\x0cGetTaskUnary\x12\x17.researcher.TaskRequest\x1a\x18.researcher.TaskResponse\"\x00\x30\x01\x12:\n\tReplyTask\x12\x16.researcher.TaskResult\x1a\x11.researcher.Empty\"\x00(\x01\x12<\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12\x1b.researcher.FeedbackMessage\x1a\x11.researcher.Empty\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'fedbiomed.transport.protocols.researcher_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FEEDBACKMESSAGE_SCALAR_METRICENTRY._options = None
  _FEEDBACKMESSAGE_SCALAR_METRICENTRY._serialized_options = b'8\001'
  _globals['_EMPTY']._serialized_start=62
  _globals['_EMPTY']._serialized_end=69
  _globals['_PROTOCOLVERSION']._serialized_start=71
  _globals['_PROTOCOLVERSION']._serialized_end=114
  _globals['_FEEDBACKMESSAGE']._serialized_start=117
  _globals['_FEEDBACKMESSAGE']._serialized_end=798
  _globals['_FEEDBACKMESSAGE_SCALAR']._serialized_start=293
  _globals['_FEEDBACKMESSAGE_SCALAR']._serialized_end=711
  _globals['_FEEDBACKMESSAGE_SCALAR_METRICENTRY']._serialized_start=632
  _globals['_FEEDBACKMESSAGE_SCALAR_METRICENTRY']._serialized_end=677
  _globals['_FEEDBACKMESSAGE_LOG']._serialized_start=713
  _globals['_FEEDBACKMESSAGE_LOG']._serialized_end=763
  _globals['_TASKREQUEST']._serialized_start=800
  _globals['_TASKREQUEST']._serialized_end=853
  _globals['_TASKRESPONSE']._serialized_start=855
  _globals['_TASKRESPONSE']._serialized_end=918
  _globals['_TASKRESULT']._serialized_start=920
  _globals['_TASKRESULT']._serialized_end=981
  _globals['_TASKRESPONSEUNARY']._serialized_start=983
  _globals['_TASKRESPONSEUNARY']._serialized_end=1018
  _globals['_RESEARCHERSERVICE']._serialized_start=1021
  _globals['_RESEARCHERSERVICE']._serialized_end=1301
# @@protoc_insertion_point(module_scope)