# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fedbiomed/proto/researcher.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n fedbiomed/proto/researcher.proto\"\x07\n\x05\x45mpty\"\xa1\x01\n\x0f\x46\x65\x65\x64\x62\x61\x63kMessage\x12,\n\x06scalar\x18\x01 \x01(\x0b\x32\x1a.FeedbackMessage.AddScalarH\x00\x12#\n\x03log\x18\x02 \x01(\x0b\x32\x14.FeedbackMessage.LogH\x00\x1a\x1b\n\tAddScalar\x12\x0e\n\x06\x62ytes_\x18\x01 \x01(\x0c\x1a\x12\n\x03Log\x12\x0b\n\x03log\x18\x01 \x01(\tB\n\n\x08\x66\x65\x65\x64\x62\x61\x63k\"\x1b\n\x0bTaskRequest\x12\x0c\n\x04node\x18\x01 \x01(\t\"?\n\x0cTaskResponse\x12\x0c\n\x04size\x18\x01 \x01(\x05\x12\x11\n\titeration\x18\x02 \x01(\x05\x12\x0e\n\x06\x62ytes_\x18\x03 \x01(\x0c\"#\n\x11TaskResponseUnary\x12\x0e\n\x06\x62ytes_\x18\x01 \x01(\x0c\x32\x9a\x01\n\x11ResearcherService\x12,\n\x07GetTask\x12\x0c.TaskRequest\x1a\r.TaskResponse\"\x00(\x01\x30\x01\x12/\n\x0cGetTaskUnary\x12\x0c.TaskRequest\x1a\r.TaskResponse\"\x00\x30\x01\x12&\n\x08\x46\x65\x65\x64\x62\x61\x63k\x12\x10.FeedbackMessage\x1a\x06.Empty\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'fedbiomed.proto.researcher_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_EMPTY']._serialized_start=36
  _globals['_EMPTY']._serialized_end=43
  _globals['_FEEDBACKMESSAGE']._serialized_start=46
  _globals['_FEEDBACKMESSAGE']._serialized_end=207
  _globals['_FEEDBACKMESSAGE_ADDSCALAR']._serialized_start=148
  _globals['_FEEDBACKMESSAGE_ADDSCALAR']._serialized_end=175
  _globals['_FEEDBACKMESSAGE_LOG']._serialized_start=177
  _globals['_FEEDBACKMESSAGE_LOG']._serialized_end=195
  _globals['_TASKREQUEST']._serialized_start=209
  _globals['_TASKREQUEST']._serialized_end=236
  _globals['_TASKRESPONSE']._serialized_start=238
  _globals['_TASKRESPONSE']._serialized_end=301
  _globals['_TASKRESPONSEUNARY']._serialized_start=303
  _globals['_TASKRESPONSEUNARY']._serialized_end=338
  _globals['_RESEARCHERSERVICE']._serialized_start=341
  _globals['_RESEARCHERSERVICE']._serialized_end=495
# @@protoc_insertion_point(module_scope)
