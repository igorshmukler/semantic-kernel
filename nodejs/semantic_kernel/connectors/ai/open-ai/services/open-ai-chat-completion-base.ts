import { AnnotationContent } from '../../../../contents/annotation-content'
import { ChatHistory } from '../../../../contents/chat-history'
import { ChatMessageContent } from '../../../../contents/chat-message-content'
import { FileReferenceContent } from '../../../../contents/file-reference-content'
import { FunctionCallContent } from '../../../../contents/function-call-content'
import { StreamingChatMessageContent } from '../../../../contents/streaming-chat-message-content'
import { StreamingTextContent } from '../../../../contents/streaming-text-content'
import { TextContent } from '../../../../contents/text-content'
import { AuthorRole } from '../../../../contents/utils/author-role'
import { FinishReason } from '../../../../contents/utils/finish-reason'
import { AutoFunctionInvocationContext } from '../../../../filters/auto-function-invocation/auto-function-invocation-context'
import { KernelArguments } from '../../../../functions/kernel-arguments'
import { Kernel } from '../../../../kernel'
import { PromptExecutionSettings } from '../../../../services/ai-service-client-base'
import { CompletionUsage } from '../../completion-usage'
import { FunctionCallChoiceConfiguration } from '../../function-call-choice-configuration'
import { updateSettingsFromFunctionCallConfiguration } from '../../function-calling-utils'
import { FunctionChoiceBehavior } from '../../function-choice-behavior'
import { FunctionChoiceType } from '../../function-choice-type'
import { OpenAIChatPromptExecutionSettings } from '../prompt-execution-settings/open-ai-prompt-execution-settings'
import { OpenAIHandler } from './open-ai-handler'

/**
 * OpenAI chat completion base class.
 */
export abstract class OpenAIChatCompletionBase extends OpenAIHandler {
  static readonly MODEL_PROVIDER_NAME = 'openai'
  static readonly SUPPORTS_FUNCTION_CALLING = true

  aiModelId: string = ''
  instructionRole: string = 'system'

  // #region Overriding base class methods

  /**
   * Get the prompt execution settings class.
   */
  getPromptExecutionSettingsClass(): any {
    return OpenAIChatPromptExecutionSettings
  }

  /**
   * Get prompt execution settings from generic settings.
   */
  getPromptExecutionSettingsFromSettings(settings: PromptExecutionSettings): OpenAIChatPromptExecutionSettings {
    if (settings instanceof OpenAIChatPromptExecutionSettings) {
      return settings
    }
    return new OpenAIChatPromptExecutionSettings(settings)
  }

  /**
   * Get the service URL.
   */
  serviceUrl(): string | null {
    return this.client.baseURL
  }

  /**
   * Internal method to get chat message contents.
   *
   * @param chatHistory - The chat history
   * @param settings - The prompt execution settings
   * @returns A promise resolving to an array of chat message contents
   */
  protected async _innerGetChatMessageContents(
    chatHistory: ChatHistory,
    settings: PromptExecutionSettings
  ): Promise<ChatMessageContent[]> {
    if (!(settings instanceof OpenAIChatPromptExecutionSettings)) {
      settings = this.getPromptExecutionSettingsFromSettings(settings)
    }

    const openAISettings = settings as OpenAIChatPromptExecutionSettings
    openAISettings.stream = false
    ;(openAISettings as any).messages = this._prepareChatHistoryForRequest(chatHistory)
    openAISettings.aiModelId = openAISettings.aiModelId || this.aiModelId

    const response = await this.sendRequest(openAISettings)

    if (!this._isChatCompletion(response)) {
      throw new Error('Expected a ChatCompletion response.')
    }

    const responseMetadata = this._getMetadataFromChatResponse(response)
    return (response as any).choices.map((choice: any) =>
      this._createChatMessageContent(response, choice, responseMetadata)
    )
  }

  /**
   * Internal method to get streaming chat message contents.
   *
   * @param chatHistory - The chat history
   * @param settings - The prompt execution settings
   * @param functionInvokeAttempt - The function invocation attempt count
   * @returns An async generator yielding streaming chat message contents
   */
  protected async *_innerGetStreamingChatMessageContents(
    chatHistory: ChatHistory,
    settings: PromptExecutionSettings,
    functionInvokeAttempt: number = 0
  ): AsyncGenerator<StreamingChatMessageContent[], void> {
    if (!(settings instanceof OpenAIChatPromptExecutionSettings)) {
      settings = this.getPromptExecutionSettingsFromSettings(settings)
    }

    const openAISettings = settings as OpenAIChatPromptExecutionSettings
    openAISettings.stream = true
    ;(openAISettings as any).streamOptions = { includeUsage: true }
    ;(openAISettings as any).messages = this._prepareChatHistoryForRequest(chatHistory)
    openAISettings.aiModelId = openAISettings.aiModelId || this.aiModelId

    const response = await this.sendRequest(openAISettings)

    if (!this._isStream(response)) {
      throw new Error('Expected an AsyncStream[ChatCompletionChunk] response.')
    }

    for await (const chunk of response as any) {
      if (chunk.choices.length === 0 && !chunk.usage) {
        continue
      }

      const chunkMetadata = this._getMetadataFromStreamingChatResponse(chunk)

      if ((!chunk.choices || chunk.choices.length === 0) && chunk.usage) {
        // Usage is contained in the last chunk where the choices are empty
        // We are duplicating the usage metadata to all the choices in the response
        yield Array.from(
          { length: openAISettings.numberOfResponses || 1 },
          (_, i) =>
            new StreamingChatMessageContent({
              role: AuthorRole.ASSISTANT,
              content: '',
              choiceIndex: i,
              innerContent: chunk,
              aiModelId: openAISettings.aiModelId,
              metadata: chunkMetadata,
              functionInvokeAttempt,
            })
        )
      } else {
        yield chunk.choices.map((choice: any) =>
          this._createStreamingChatMessageContent(chunk, choice, chunkMetadata, functionInvokeAttempt)
        )
      }
    }
  }

  /**
   * Verify function choice settings.
   *
   * @param settings - The prompt execution settings to verify
   */
  protected _verifyFunctionChoiceSettings(settings: PromptExecutionSettings): void {
    if (!(settings instanceof OpenAIChatPromptExecutionSettings)) {
      throw new Error('The settings must be an OpenAIChatPromptExecutionSettings.')
    }

    if (settings.numberOfResponses && settings.numberOfResponses > 1) {
      throw new Error(
        'Auto-invocation of tool calls may only be used with a ' + 'OpenAIChatPromptExecutions.numberOfResponses of 1.'
      )
    }
  }

  /**
   * Get the update function choice settings callback.
   */
  protected _updateFunctionChoiceSettingsCallback(): (
    config: FunctionCallChoiceConfiguration,
    settings: PromptExecutionSettings,
    choiceType: FunctionChoiceType
  ) => void {
    return updateSettingsFromFunctionCallConfiguration
  }

  /**
   * Reset function choice settings.
   *
   * @param settings - The prompt execution settings to reset
   */
  protected _resetFunctionChoiceSettings(settings: PromptExecutionSettings): void {
    const anySettings = settings as any
    if ('toolChoice' in anySettings) {
      anySettings.toolChoice = null
    }
    if ('tools' in anySettings) {
      anySettings.tools = null
    }
  }

  // #endregion

  // #region Content creation

  /**
   * Create a chat message content object from a choice.
   *
   * @param response - The chat completion response
   * @param choice - The choice to convert
   * @param responseMetadata - Response metadata
   * @returns A chat message content
   */
  protected _createChatMessageContent(
    response: any,
    choice: any,
    responseMetadata: Record<string, any>
  ): ChatMessageContent {
    const metadata = { ...this._getMetadataFromChatChoice(choice), ...responseMetadata }

    const items: any[] = [...this._getToolCallsFromChatChoice(choice), ...this._getFunctionCallFromChatChoice(choice)]

    if (choice.message.content) {
      items.push(new TextContent({ text: choice.message.content }))
    } else if ('refusal' in choice.message && choice.message.refusal) {
      items.push(new TextContent({ text: choice.message.refusal }))
    }

    return new ChatMessageContent({
      innerContent: response,
      aiModelId: this.aiModelId,
      metadata,
      role: AuthorRole[choice.message.role.toUpperCase() as keyof typeof AuthorRole] || AuthorRole.ASSISTANT,
      items,
      finishReason: choice.finish_reason
        ? FinishReason[choice.finish_reason.toUpperCase() as keyof typeof FinishReason]
        : undefined,
    })
  }

  /**
   * Create a streaming chat message content object from a choice.
   *
   * @param chunk - The chat completion chunk
   * @param choice - The choice to convert
   * @param chunkMetadata - Chunk metadata
   * @param functionInvokeAttempt - The function invocation attempt count
   * @returns A streaming chat message content
   */
  protected _createStreamingChatMessageContent(
    chunk: any,
    choice: any,
    chunkMetadata: Record<string, any>,
    functionInvokeAttempt: number
  ): StreamingChatMessageContent {
    const metadata = { ...this._getMetadataFromChatChoice(choice), ...chunkMetadata }

    const items: any[] = [...this._getToolCallsFromChatChoice(choice), ...this._getFunctionCallFromChatChoice(choice)]

    if (choice.delta?.content) {
      items.push(
        new StreamingTextContent({
          choiceIndex: choice.index,
          text: choice.delta.content,
        })
      )
    }

    return new StreamingChatMessageContent({
      choiceIndex: choice.index,
      innerContent: chunk,
      aiModelId: this.aiModelId,
      metadata,
      role: choice.delta?.role
        ? AuthorRole[choice.delta.role.toUpperCase() as keyof typeof AuthorRole]
        : AuthorRole.ASSISTANT,
      finishReason: choice.finish_reason
        ? FinishReason[choice.finish_reason.toUpperCase() as keyof typeof FinishReason]
        : undefined,
      items,
      functionInvokeAttempt,
    })
  }

  /**
   * Get metadata from a chat response.
   *
   * @param response - The chat completion response
   * @returns Metadata object
   */
  protected _getMetadataFromChatResponse(response: any): Record<string, any> {
    return {
      id: response.id,
      created: response.created,
      systemFingerprint: response.system_fingerprint,
      usage: response.usage ? CompletionUsage.fromOpenAI(response.usage) : null,
    }
  }

  /**
   * Get metadata from a streaming chat response.
   *
   * @param response - The chat completion chunk
   * @returns Metadata object
   */
  protected _getMetadataFromStreamingChatResponse(response: any): Record<string, any> {
    return {
      id: response.id,
      created: response.created,
      systemFingerprint: response.system_fingerprint,
      usage: response.usage ? CompletionUsage.fromOpenAI(response.usage) : null,
    }
  }

  /**
   * Get metadata from a chat choice.
   *
   * @param choice - The choice
   * @returns Metadata object
   */
  protected _getMetadataFromChatChoice(choice: any): Record<string, any> {
    return {
      logprobs: (choice as any).logprobs || null,
    }
  }

  /**
   * Get tool calls from a chat choice.
   *
   * @param choice - The choice
   * @returns Array of function call contents
   */
  protected _getToolCallsFromChatChoice(choice: any): FunctionCallContent[] {
    const content = 'message' in choice ? choice.message : choice.delta

    if (content && 'tool_calls' in content && content.tool_calls) {
      return content.tool_calls
        .filter((tool: any) => tool.function)
        .map(
          (tool: any) =>
            new FunctionCallContent({
              id: tool.id,
              index: tool.index,
              name: tool.function.name,
              arguments: tool.function.arguments,
            })
        )
    }

    // When you enable asynchronous content filtering in Azure OpenAI, you may receive empty deltas
    return []
  }

  /**
   * Get function call from a chat choice (legacy).
   *
   * @param choice - The choice
   * @returns Array of function call contents
   */
  protected _getFunctionCallFromChatChoice(choice: any): FunctionCallContent[] {
    const content = 'message' in choice ? choice.message : choice.delta

    if (content && 'function_call' in content && content.function_call) {
      const functionCall = content.function_call
      return [
        new FunctionCallContent({
          id: 'legacy_function_call',
          name: functionCall.name || '',
          arguments: functionCall.arguments || '',
        }),
      ]
    }

    // When you enable asynchronous content filtering in Azure OpenAI, you may receive empty deltas
    return []
  }

  /**
   * Prepare the chat history for a request.
   *
   * Allowing customization of the key names for role/author, and optionally overriding the role.
   *
   * ChatRole.TOOL messages need to be formatted different than system/user/assistant messages:
   * They require a "tool_call_id" and (function) "name" key, and the "metadata" key should
   * be removed. The "encoding" key should also be removed.
   *
   * @param chatHistory - The chat history to prepare
   * @param roleKey - The key name for the role/author
   * @param contentKey - The key name for the content/message
   * @returns The prepared chat history for a request
   */
  protected _prepareChatHistoryForRequest(
    chatHistory: ChatHistory,
    roleKey: string = 'role',
    contentKey: string = 'content'
  ): any[] {
    return chatHistory.messages
      .filter((message) => !(message instanceof AnnotationContent) && !(message instanceof FileReferenceContent))
      .map((message) => {
        const dict = message.toDict(roleKey, contentKey)

        // Override system role with developer if instructionRole is set to 'developer'
        if (this.instructionRole === 'developer' && dict[roleKey] === 'system') {
          return {
            ...dict,
            [roleKey]: 'developer',
          }
        }

        return dict
      })
  }

  // #endregion

  // #region Function calling

  /**
   * Process a function call (deprecated - use kernel.invokeFunction instead).
   *
   * @deprecated Use `invoke_function_call` from the kernel instead with `FunctionChoiceBehavior`.
   */
  protected async _processFunctionCall(
    _functionCall: FunctionCallContent,
    _chatHistory: ChatHistory,
    _kernel: Kernel,
    _args: KernelArguments | null,
    _functionCallCount: number,
    _requestIndex: number,
    _functionCallBehavior: FunctionChoiceBehavior
  ): Promise<AutoFunctionInvocationContext | null> {
    // This method should delegate to the kernel's function invocation logic
    // The exact implementation depends on the kernel's API
    return null
  }

  // #endregion

  // #region Type guards

  /**
   * Type guard to check if response is a ChatCompletion.
   */
  private _isChatCompletion(response: any): boolean {
    return response && 'choices' in response && !(Symbol.asyncIterator in response)
  }

  /**
   * Type guard to check if response is a Stream.
   */
  private _isStream(response: any): boolean {
    return response && Symbol.asyncIterator in response
  }

  // #endregion
}
