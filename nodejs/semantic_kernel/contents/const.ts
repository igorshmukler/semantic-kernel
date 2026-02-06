/**
 * Content type tags and constants.
 */

export const AUDIO_CONTENT_TAG = 'audio'
export const CHAT_MESSAGE_CONTENT_TAG = 'message'
export const CHAT_HISTORY_TAG = 'chat_history'
export const TEXT_CONTENT_TAG = 'text'
export const IMAGE_CONTENT_TAG = 'image'
export const ANNOTATION_CONTENT_TAG = 'annotation'
export const STREAMING_ANNOTATION_CONTENT_TAG = 'streaming_annotation'
export const BINARY_CONTENT_TAG = 'binary'
export const FILE_REFERENCE_CONTENT_TAG = 'file_reference'
export const STREAMING_FILE_REFERENCE_CONTENT_TAG = 'streaming_file_reference'
export const FUNCTION_CALL_CONTENT_TAG = 'function_call'
export const FUNCTION_RESULT_CONTENT_TAG = 'function_result'
export const REASONING_CONTENT_TAG = 'reasoning'
export const DISCRIMINATOR_FIELD = 'content_type'

/**
 * Content types enumeration.
 */
export enum ContentTypes {
  AUDIO_CONTENT = 'audio',
  ANNOTATION_CONTENT = 'annotation',
  BINARY_CONTENT = 'binary',
  CHAT_MESSAGE_CONTENT = 'message',
  IMAGE_CONTENT = 'image',
  FILE_REFERENCE_CONTENT = 'file_reference',
  FUNCTION_CALL_CONTENT = 'function_call',
  FUNCTION_RESULT_CONTENT = 'function_result',
  REASONING_CONTENT = 'reasoning',
  STREAMING_ANNOTATION_CONTENT = 'streaming_annotation',
  STREAMING_FILE_REFERENCE_CONTENT = 'streaming_file_reference',
  TEXT_CONTENT = 'text',
}
