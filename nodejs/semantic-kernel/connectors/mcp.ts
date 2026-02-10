// MCP plugin base and implementations for stdio, SSE, streamable HTTP, and websocket clients.
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { WebSocketClientTransport } from '@modelcontextprotocol/sdk/client/websocket.js'
import type {
    CallToolResult,
    CreateMessageRequest,
    CreateMessageResult,
    EmbeddedResource,
    LoggingLevel,
    AudioContent as McpAudioContent,
    ImageContent as McpImageContent,
    TextContent as McpTextContent,
    Prompt,
    PromptMessage,
    Tool,
} from '@modelcontextprotocol/sdk/types.js'
import { AudioContent } from '../contents/audio-content'
import { BinaryContent } from '../contents/binary-content'
import { ChatHistory } from '../contents/chat-history'
import { ChatMessageContent } from '../contents/chat-message-content'
import { ImageContent } from '../contents/image-content'
import { TextContent } from '../contents/text-content'
import { AuthorRole } from '../contents/utils/author-role'
import { Kernel } from '../kernel'
import { createDefaultLogger } from '../utils/logger'
import { ChatCompletionClientBase } from './ai/chat-completion-client-base'

/**
 * Generic binary content implementation for MCP resources
 */
class GenericBinaryContent extends BinaryContent {
  protected defaultMimeType: string = 'application/octet-stream'
  readonly tag: string = 'binary'
}

// Type aliases for MCP SDK types
type McpPromptMessage = PromptMessage
type McpSamplingMessage = PromptMessage
type McpCallToolResult = CallToolResult
type McpTool = Tool
type McpPrompt = Prompt
type McpEmbeddedResource = EmbeddedResource

const logger = createDefaultLogger('MCP')

const INTERNAL_ERROR = -1 // MCP error code constant

// Map MCP logging levels to logger levels
const LOG_LEVEL_MAPPING: Record<LoggingLevel, string> = {
  debug: 'debug',
  info: 'info',
  notice: 'info',
  warning: 'warn',
  error: 'error',
  critical: 'error',
  alert: 'error',
  emergency: 'error',
}

// #region: Helper Functions

/**
 * Convert MCP PromptMessage to Kernel ChatMessageContent
 */
export function mcpPromptMessageToKernelContent(mcpMsg: McpPromptMessage): ChatMessageContent {
  // Handle content as array
  const contentItems = Array.isArray(mcpMsg.content) ? mcpMsg.content : [mcpMsg.content]
  const kernelContent = contentItems.map((item) => mcpContentTypesToKernelContent(item as any))

  // Map string role to AuthorRole enum
  const roleMap: Record<string, AuthorRole> = {
    system: AuthorRole.SYSTEM,
    user: AuthorRole.USER,
    assistant: AuthorRole.ASSISTANT,
    tool: AuthorRole.TOOL,
    developer: AuthorRole.DEVELOPER,
  }
  const role = roleMap[mcpMsg.role.toLowerCase()] || AuthorRole.USER

  return new ChatMessageContent({
    role,
    items: kernelContent,
    innerContent: mcpMsg,
  })
}

/**
 * Convert MCP CallToolResult to Kernel content list
 */
export function mcpCallToolResultToKernelContents(
  mcpResult: McpCallToolResult
): Array<TextContent | ImageContent | BinaryContent | AudioContent> {
  return mcpResult.content.map((item) => mcpContentTypesToKernelContent(item as any))
}

/**
 * Convert MCP content types to Kernel content
 */
export function mcpContentTypesToKernelContent(
  mcpType: McpTextContent | McpImageContent | McpAudioContent | McpEmbeddedResource | any
): TextContent | ImageContent | BinaryContent | AudioContent {
  // TextContent
  if (mcpType.type === 'text' && 'text' in mcpType) {
    return new TextContent({
      text: mcpType.text,
      innerContent: mcpType,
    })
  }

  // ImageContent
  if (mcpType.type === 'image' && 'data' in mcpType && 'mimeType' in mcpType) {
    return new ImageContent({
      data: mcpType.data,
      mimeType: mcpType.mimeType,
      innerContent: mcpType,
    })
  }

  // AudioContent
  if (mcpType.type === 'audio' && 'data' in mcpType && 'mimeType' in mcpType) {
    return new AudioContent({
      data: mcpType.data,
      mimeType: mcpType.mimeType,
      innerContent: mcpType,
    })
  }

  // ResourceLink
  if (mcpType.type === 'resource_link' && 'uri' in mcpType) {
    return new GenericBinaryContent({
      uri: mcpType.uri,
      mimeType: mcpType.mimeType || 'application/octet-stream',
      innerContent: mcpType,
    })
  }

  // EmbeddedResource
  if (mcpType.type === 'resource' && 'resource' in mcpType) {
    const resource = mcpType.resource
    if ('text' in resource) {
      return new TextContent({
        text: resource.text,
        innerContent: mcpType,
        metadata: mcpType.annotations || {},
      })
    }
    if ('blob' in resource) {
      return new GenericBinaryContent({
        data: resource.blob,
        mimeType: resource.mimeType || 'application/octet-stream',
        innerContent: mcpType,
        metadata: mcpType.annotations || {},
      })
    }
  }

  // Fallback
  return new GenericBinaryContent({
    data: '',
    innerContent: mcpType,
  })
}

/**
 * Convert Kernel content to MCP content types
 */
export function kernelContentToMcpContentTypes(
  content: TextContent | ImageContent | BinaryContent | AudioContent | ChatMessageContent
): Array<McpTextContent | McpImageContent | McpAudioContent | McpEmbeddedResource> {
  // TextContent
  if (content instanceof TextContent || ('text' in content && !('items' in content))) {
    return [{ type: 'text', text: (content as TextContent).text }]
  }

  // ImageContent
  if (content instanceof ImageContent || ('data' in content && 'mimeType' in content && !('uri' in content))) {
    const imgContent = content as ImageContent
    return [
      {
        type: 'image',
        data: imgContent.dataString,
        mimeType: imgContent.mimeType,
      },
    ]
  }

  // AudioContent - check for audio type
  if (content instanceof AudioContent || ((content as any).type === 'audio' && 'data' in content)) {
    const audioContent = content as AudioContent
    return [
      {
        type: 'audio',
        data: audioContent.dataString,
        mimeType: audioContent.mimeType,
      },
    ]
  }

  // BinaryContent
  if (content instanceof BinaryContent || ('data' in content && 'uri' in content)) {
    const binContent = content as BinaryContent
    return [
      {
        type: 'resource',
        resource: {
          blob: binContent.dataString,
          mimeType: binContent.mimeType,
          uri: binContent.uri || 'sk://binary',
        },
      } as McpEmbeddedResource,
    ]
  }

  // ChatMessageContent
  if (content instanceof ChatMessageContent || 'items' in content) {
    const messages: Array<McpTextContent | McpImageContent | McpAudioContent | McpEmbeddedResource> = []
    for (const item of (content as ChatMessageContent).items) {
      // Only process supported content types (matching Python implementation)
      if (
        item instanceof TextContent ||
        item instanceof ImageContent ||
        item instanceof BinaryContent ||
        item instanceof AudioContent
      ) {
        messages.push(...kernelContentToMcpContentTypes(item))
      } else {
        logger.debug('Unsupported content type:', typeof item)
      }
    }
    return messages
  }

  throw new Error(`Unsupported content type: ${typeof content}`)
}

/**
 * Get parameter dictionary from MCP prompt
 */
function getParameterDictFromMcpPrompt(prompt: McpPrompt): Array<Record<string, any>> {
  if (!prompt.arguments) {
    return []
  }

  return prompt.arguments.map((arg) => ({
    name: arg.name,
    description: arg.description,
    is_required: arg.required !== false,
    type_object: String,
  }))
}

/**
 * Get parameter dictionaries from MCP tool
 */
function getParameterDictsFromMcpTool(tool: McpTool): Array<Record<string, any>> {
  const properties = tool.inputSchema.properties
  const required = (tool.inputSchema.required as string[]) || []

  if (!properties) {
    return []
  }

  const params: Array<Record<string, any>> = []
  for (const [propName, propDetails] of Object.entries(properties)) {
    const details = typeof propDetails === 'string' ? JSON.parse(propDetails) : propDetails

    params.push({
      name: propName,
      is_required: required.includes(propName),
      type: details.type,
      default_value: details.default ?? null,
      schema_data: details,
    })
  }

  return params
}

/**
 * Normalize MCP tool/prompt names to allowed identifier pattern
 */
function normalizeMcpName(name: string): string {
  return name.replace(/[^A-Za-z0-9_.-]/g, '-')
}

// #endregion

// #region: MCP Plugin Base

export interface MCPPluginBaseOptions {
  name: string
  description?: string
  loadTools?: boolean
  loadPrompts?: boolean
  client?: Client
  kernel?: Kernel
  requestTimeout?: number
}

export abstract class MCPPluginBase {
  name: string
  description?: string
  loadToolsFlag: boolean
  loadPromptsFlag: boolean
  client?: Client
  kernel?: Kernel
  requestTimeout?: number

  private exitStack: Array<() => Promise<void>> = []
  private currentTask?: Promise<void>
  private stopEvent?: { wait: () => Promise<void>; set: () => void }
  private transport?: StdioClientTransport | SSEClientTransport | WebSocketClientTransport

  constructor(options: MCPPluginBaseOptions) {
    this.name = options.name
    this.description = options.description
    this.loadToolsFlag = options.loadTools ?? true
    this.loadPromptsFlag = options.loadPrompts ?? true
    this.client = options.client
    this.kernel = options.kernel
    this.requestTimeout = options.requestTimeout
  }

  /**
   * Context manager entry
   */
  async __aenter__(): Promise<this> {
    await this.connect()
    return this
  }

  /**
   * Context manager exit
   */
  async __aexit__(_excType?: any, _excValue?: any, _traceback?: any): Promise<void> {
    await this.close()
  }

  /**
   * Connect to the MCP server
   */
  async connect(): Promise<void> {
    const readyEvent = this.createEvent()

    try {
      this.currentTask = this.innerConnect(readyEvent)
      await readyEvent.wait()
    } catch (error) {
      readyEvent.set()
      if (error instanceof Error && error.message.includes('Invalid configuration')) {
        throw error
      }
      await this.close()
      throw new Error(`Failed to enter context manager: ${error}`)
    }
  }

  /**
   * Disconnect from the MCP server
   */
  async close(): Promise<void> {
    if (this.stopEvent) {
      this.stopEvent.set()
    }
    if (this.currentTask) {
      await this.currentTask
      this.currentTask = undefined
    }
    if (this.client) {
      await this.client.close()
      this.client = undefined
    }
    if (this.transport) {
      await this.transport.close()
      this.transport = undefined
    }
  }

  /**
   * Internal connection logic
   */
  private async innerConnect(readyEvent: { wait: () => Promise<void>; set: () => void }): Promise<void> {
    if (!this.client) {
      try {
        this.transport = await this.createTransport()
      } catch (error) {
        await this.closeExitStack()
        readyEvent.set()
        throw new Error(`Failed to create transport. Please check your configuration: ${error}`)
      }

      try {
        // Create client with transport and optional timeout
        const clientOptions: any = {
          capabilities: {},
        }

        // Add request timeout if specified (in milliseconds)
        if (this.requestTimeout !== undefined) {
          clientOptions.timeout = this.requestTimeout * 1000 // Convert seconds to milliseconds
        }

        this.client = new Client(
          {
            name: this.name,
            version: '1.0.0',
          },
          clientOptions
        )
        await this.client.connect(this.transport)
      } catch (error) {
        await this.closeExitStack()
        throw new Error(`Failed to connect client. Please check your configuration: ${error}`)
      }
    } else {
      // Check if client exists but needs reinitialization
      // The MCP SDK Client doesn't expose request_id directly, but we can check if it's connected
      // If the client exists but is not properly initialized, reconnect it
      if (this.transport && !(this.client as any)._initialized) {
        try {
          await this.client.connect(this.transport)
        } catch (error) {
          logger.warn('Failed to reinitialize client:', error)
        }
      }
    }

    logger.debug(`Connected to MCP server: ${this.name}`)

    if (this.loadToolsFlag) {
      await this.loadTools()
    }
    if (this.loadPromptsFlag) {
      await this.loadPrompts()
    }

    // Signal ready
    readyEvent.set()

    // Wait for stop signal
    this.stopEvent = this.createEvent()
    await this.stopEvent.wait()

    try {
      await this.closeExitStack()
    } catch (error) {
      logger.error('Error during exit stack close', error)
    }
  }

  /**
   * Sampling callback for AI completion requests
   * Called when the MCP server needs to get a message completed
   */
  async samplingCallback(
    params: CreateMessageRequest['params']
  ): Promise<CreateMessageResult | { code: number; message: string }> {
    if (!this.kernel || !this.kernel.services || this.kernel.services.size === 0) {
      return {
        code: INTERNAL_ERROR,
        message: 'No services in Kernel. Please set a kernel with one or more services.',
      }
    }

    logger.debug('Sampling callback called with params:', params)

    // Get service names from model preferences or use default
    let names: string[] = ['default']
    if (params.modelPreferences?.hints) {
      names = params.modelPreferences.hints.map((hint: any) => hint.name || 'default')
    }

    // Try to get a chat completion service
    let service: ChatCompletionClientBase | undefined
    for (const name of names) {
      try {
        service = this.kernel.getService(name) as ChatCompletionClientBase
        if (service) break
      } catch {
        continue
      }
    }

    if (!service) {
      try {
        service = this.kernel.getService('default') as ChatCompletionClientBase
      } catch {
        // No service found
      }
    }

    if (!service) {
      return {
        code: INTERNAL_ERROR,
        message: 'No Chat completion service found.',
      }
    }

    // Create completion settings
    const completionSettings = service.instantiatePromptExecutionSettings()

    // Set temperature if supported
    if (params.temperature !== undefined) {
      ;(completionSettings as any).temperature = params.temperature
    }

    // Set max tokens if supported (try different property names)
    if (params.maxTokens !== undefined) {
      if ('maxCompletionTokens' in completionSettings) {
        ;(completionSettings as any).maxCompletionTokens = params.maxTokens
      } else if ('maxTokens' in completionSettings) {
        ;(completionSettings as any).maxTokens = params.maxTokens
      } else if ('maxOutputTokens' in completionSettings) {
        ;(completionSettings as any).maxOutputTokens = params.maxTokens
      }
    }

    // Create chat history
    const chatHistory = new ChatHistory([], params.systemPrompt)
    for (const msg of params.messages) {
      chatHistory.addMessage(mcpPromptMessageToKernelContent(msg as PromptMessage))
    }

    // Get chat message content
    let result: ChatMessageContent
    try {
      const results = await service.getChatMessageContents(chatHistory, completionSettings, { kernel: this.kernel })
      if (!results || results.length === 0) {
        return {
          code: INTERNAL_ERROR,
          message: 'Failed to get chat message content.',
        }
      }
      result = results[0]
    } catch (error) {
      return {
        code: INTERNAL_ERROR,
        message: `Failed to get chat message content: ${error}`,
      }
    }

    // Convert result to MCP content
    const mcpContents = kernelContentToMcpContentTypes(result)

    // Find first text or image content
    const mcpContent = mcpContents.find((content) => content.type === 'text' || content.type === 'image')

    if (!mcpContent) {
      return {
        code: INTERNAL_ERROR,
        message: 'Failed to get right content types from the response.',
      }
    }

    return {
      role: 'assistant',
      content: mcpContent,
      model: service.aiModelId,
    }
  }

  /**
   * Logging callback for MCP server log messages
   * Called when the MCP server sends a log message
   * Override this method in subclasses to customize logging behavior
   */
  async loggingCallback(params: { level: LoggingLevel; data: any; logger?: string }): Promise<void> {
    const logLevel = LOG_LEVEL_MAPPING[params.level] || 'info'
    const loggerInstance = logger as any

    // Call the appropriate log method on the logger
    if (typeof loggerInstance[logLevel] === 'function') {
      loggerInstance[logLevel](params.data)
    } else {
      logger.info(params.data)
    }
  }

  /**
   * Message handler for MCP server notifications
   */
  async messageHandler(notification: { method: string; params?: any }): Promise<void> {
    switch (notification.method) {
      case 'notifications/tools/list_changed':
        await this.loadTools()
        break
      case 'notifications/prompts/list_changed':
        await this.loadPrompts()
        break
      default:
        logger.debug('Unhandled notification:', notification.method)
    }
  }

  /**
   * Load prompts from MCP server
   */
  async loadPrompts(): Promise<void> {
    try {
      const promptList = await this.client!.listPrompts()

      for (const prompt of promptList.prompts || []) {
        const localName = normalizeMcpName(prompt.name)
        const func = this.createPromptFunction(prompt.name, prompt.description || '')
        const params = getParameterDictFromMcpPrompt(prompt)

        // Attach metadata to function
        ;(func as any).__kernel_function_name__ = localName
        ;(func as any).__kernel_function_description__ = prompt.description
        ;(func as any).__kernel_function_parameters__ = params

        // Attach to this instance
        ;(this as any)[localName] = func
      }
    } catch (error) {
      logger.warn('Failed to load prompts:', error)
    }
  }

  /**
   * Load tools from MCP server
   */
  async loadTools(): Promise<void> {
    try {
      const toolList = await this.client!.listTools()

      for (const tool of toolList.tools || []) {
        const localName = normalizeMcpName(tool.name)
        const func = this.createToolFunction(tool.name, tool.description || '')
        const params = getParameterDictsFromMcpTool(tool)

        // Attach metadata to function
        ;(func as any).__kernel_function_name__ = localName
        ;(func as any).__kernel_function_description__ = tool.description
        ;(func as any).__kernel_function_parameters__ = params

        // Attach to this instance
        ;(this as any)[localName] = func
      }
    } catch (error) {
      logger.warn('Failed to load tools:', error)
    }
  }

  /**
   * Call a tool with given arguments
   */
  async callTool(
    toolName: string,
    kwargs: Record<string, any> = {}
  ): Promise<Array<TextContent | ImageContent | BinaryContent>> {
    if (!this.client) {
      throw new Error('MCP server not connected, please call connect() before using this method.')
    }
    if (!this.loadToolsFlag) {
      throw new Error('Tools are not loaded for this server, please set loadTools=true in the constructor.')
    }

    try {
      const result = await this.client.callTool({ name: toolName, arguments: kwargs })
      // Check if result has content (successful call) or is an error
      if ('content' in result && result.content) {
        return mcpCallToolResultToKernelContents(result as McpCallToolResult)
      }
      throw new Error(`Tool call returned no content: ${JSON.stringify(result)}`)
    } catch (error) {
      throw new Error(`Failed to call tool '${toolName}': ${error}`)
    }
  }

  /**
   * Get a prompt with given arguments
   */
  async getPrompt(promptName: string, kwargs: Record<string, any> = {}): Promise<ChatMessageContent[]> {
    if (!this.client) {
      throw new Error('MCP server not connected, please call connect() before using this method.')
    }
    if (!this.loadPromptsFlag) {
      throw new Error('Prompts are not loaded for this server, please set loadPrompts=true in the constructor.')
    }

    try {
      const promptResult = await this.client.getPrompt({ name: promptName, arguments: kwargs })
      return promptResult.messages.map(mcpPromptMessageToKernelContent)
    } catch (error) {
      throw new Error(`Failed to call prompt '${promptName}': ${error}`)
    }
  }

  /**
   * Called when plugin is added to kernel
   */
  addedToKernel(kernel: Kernel): void {
    this.kernel = kernel
  }

  /**
   * Abstract method to create MCP transport
   */
  protected abstract createTransport(): Promise<StdioClientTransport | SSEClientTransport | WebSocketClientTransport>

  /**
   * Helper to create an event
   */
  private createEvent(): { wait: () => Promise<void>; set: () => void } {
    let resolve: () => void
    const promise = new Promise<void>((res) => {
      resolve = res
    })
    return {
      wait: () => promise,
      set: () => resolve(),
    }
  }

  /**
   * Close all resources in exit stack
   */
  private async closeExitStack(): Promise<void> {
    for (const closer of this.exitStack.reverse()) {
      await closer()
    }
    this.exitStack = []
  }

  /**
   * Create a function wrapper for tool calls
   */
  private createToolFunction(toolName: string, _description: string) {
    return async (kwargs: Record<string, any> = {}) => {
      return this.callTool(toolName, kwargs)
    }
  }

  /**
   * Create a function wrapper for prompt calls
   */
  private createPromptFunction(promptName: string, _description: string) {
    return async (kwargs: Record<string, any> = {}) => {
      return this.getPrompt(promptName, kwargs)
    }
  }
}

// #endregion

// #region: MCP Plugin Implementations

export interface MCPStdioPluginOptions extends MCPPluginBaseOptions {
  command: string
  args?: string[]
  env?: Record<string, string>
}

export class MCPStdioPlugin extends MCPPluginBase {
  command: string
  args: string[]
  env?: Record<string, string>

  constructor(options: MCPStdioPluginOptions) {
    super(options)
    this.command = options.command
    this.args = options.args || []
    this.env = options.env
  }

  protected async createTransport(): Promise<StdioClientTransport> {
    logger.info('Creating stdio MCP transport')
    const transport = new StdioClientTransport({
      command: this.command,
      args: this.args,
      env: this.env,
    })
    return transport
  }
}

export interface MCPSsePluginOptions extends MCPPluginBaseOptions {
  url: string
  headers?: Record<string, string>
  timeout?: number
  sseReadTimeout?: number
}

export class MCPSsePlugin extends MCPPluginBase {
  url: string
  headers?: Record<string, string>
  timeout?: number
  sseReadTimeout?: number

  constructor(options: MCPSsePluginOptions) {
    super(options)
    this.url = options.url
    this.headers = options.headers
    this.timeout = options.timeout
    this.sseReadTimeout = options.sseReadTimeout
  }

  protected async createTransport(): Promise<SSEClientTransport> {
    logger.info('Creating SSE MCP transport')
    const transport = new SSEClientTransport(new URL(this.url))
    return transport
  }
}

export interface MCPStreamableHttpPluginOptions extends MCPPluginBaseOptions {
  url: string
  headers?: Record<string, string>
  timeout?: number
  sseReadTimeout?: number
  terminateOnClose?: boolean
}

export class MCPStreamableHttpPlugin extends MCPPluginBase {
  url: string
  headers?: Record<string, string>
  timeout?: number
  sseReadTimeout?: number
  terminateOnClose?: boolean

  constructor(options: MCPStreamableHttpPluginOptions) {
    super(options)
    this.url = options.url
    this.headers = options.headers
    this.timeout = options.timeout
    this.sseReadTimeout = options.sseReadTimeout
    this.terminateOnClose = options.terminateOnClose
  }

  protected async createTransport(): Promise<SSEClientTransport> {
    logger.info('Creating Streamable HTTP MCP transport')
    // Note: Using SSEClientTransport as streamable HTTP uses SSE under the hood
    // If a dedicated HTTP transport becomes available in the SDK, update this
    const transport = new SSEClientTransport(new URL(this.url))
    return transport
  }
}

export interface MCPWebsocketPluginOptions extends MCPPluginBaseOptions {
  url: string
}

export class MCPWebsocketPlugin extends MCPPluginBase {
  url: string

  constructor(options: MCPWebsocketPluginOptions) {
    super(options)
    this.url = options.url
  }

  protected async createTransport(): Promise<WebSocketClientTransport> {
    logger.info('Creating WebSocket MCP transport')
    const transport = new WebSocketClientTransport(new URL(this.url))
    return transport
  }
}

// #endregion
