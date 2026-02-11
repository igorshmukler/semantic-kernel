import { randomUUID } from 'crypto'
import { experimental } from '../../../utils/feature-stage-decorator'
import { createDefaultLogger } from '../../../utils/logger'
import { Agent } from '../core/agent'
import type { AgentId } from '../core/agent-id'
import { CoreAgentId } from '../core/agent-id'
import { AgentMetadata } from '../core/agent-metadata'
import { AgentType, CoreAgentType } from '../core/agent-type'
import { AgentInstantiationContext } from '../core/base-agent'
import { CancellationToken } from '../core/cancellation-token'
import { CoreRuntime } from '../core/core-runtime'
import { MessageDroppedException } from '../core/exceptions'
import { DropMessage, InterventionHandler } from '../core/intervention'
import { MessageContext } from '../core/message-context'
import { JSON_DATA_CONTENT_TYPE, MessageSerializer, SerializationRegistry } from '../core/serialization'
import { Subscription } from '../core/subscription'
import { TopicId } from '../core/topic'

const logger = createDefaultLogger('InProcessRuntime')

/**
 * Utility function to check if the intervention handler returned undefined and issue a warning.
 *
 * @param value - The return value to check
 * @param handlerName - Name of the intervention handler method for the warning message
 */
function warnIfUndefined(value: any, handlerName: string): void {
  if (value === undefined) {
    logger.warn(
      `Intervention handler ${handlerName} returned undefined. This might be unintentional. ` +
        'Consider returning the original message or DropMessage explicitly.'
    )
  }
}

/**
 * Message kind enum for event logging.
 */
enum MessageKind {
  DIRECT = 'DIRECT',
  PUBLISH = 'PUBLISH',
  RESPOND = 'RESPOND',
}

/**
 * Delivery stage enum for event logging.
 */
enum DeliveryStage {
  SEND = 'SEND',
  DELIVER = 'DELIVER',
}

/**
 * Helper to create a structured message event for logging.
 */
function logMessageEvent(options: {
  payload: string
  sender: AgentId | null
  receiver: AgentId | TopicId | null
  kind: MessageKind
  deliveryStage: DeliveryStage
}): void {
  logger.info(
    JSON.stringify({
      type: 'Message',
      payload: options.payload,
      sender: options.sender ? options.sender.toString() : null,
      receiver: options.receiver ? options.receiver.toString() : null,
      kind: options.kind,
      delivery_stage: options.deliveryStage,
    })
  )
}

/**
 * Helper to create a structured message dropped event for logging.
 */
function logMessageDroppedEvent(options: {
  payload: string
  sender: AgentId | null
  receiver: AgentId | TopicId | null
  kind: MessageKind
}): void {
  logger.info(
    JSON.stringify({
      type: 'MessageDropped',
      payload: options.payload,
      sender: options.sender ? options.sender.toString() : null,
      receiver: options.receiver ? options.receiver.toString() : null,
      kind: options.kind,
    })
  )
}

/**
 * Helper to create a structured message handler exception event for logging.
 */
function logMessageHandlerExceptionEvent(options: { payload: string; handlingAgent: AgentId; exception: Error }): void {
  logger.info(
    JSON.stringify({
      type: 'MessageHandlerException',
      payload: options.payload,
      handling_agent: options.handlingAgent.toString(),
      exception: options.exception.toString(),
      exception_message: options.exception.message,
      exception_stack: options.exception.stack,
    })
  )
}

/**
 * Helper to create a structured agent construction exception event for logging.
 */
function logAgentConstructionExceptionEvent(options: { agentId: AgentId; exception: Error }): void {
  logger.info(
    JSON.stringify({
      type: 'AgentConstructionException',
      agent_id: options.agentId.toString(),
      exception: options.exception.toString(),
      exception_message: options.exception.message,
      exception_stack: options.exception.stack,
    })
  )
}

/**
 * A message envelope for publishing messages to all agents.
 */
@experimental
class PublishMessageEnvelope {
  message: any
  cancellationToken: CancellationToken
  sender: AgentId | null
  topicId: TopicId
  messageId: string

  constructor(
    message: any,
    cancellationToken: CancellationToken,
    sender: AgentId | null,
    topicId: TopicId,
    messageId: string
  ) {
    this.message = message
    this.cancellationToken = cancellationToken
    this.sender = sender
    this.topicId = topicId
    this.messageId = messageId
  }
}

/**
 * A message envelope for sending a message to a specific agent.
 */
@experimental
class SendMessageEnvelope {
  message: any
  sender: AgentId | null
  recipient: AgentId
  cancellationToken: CancellationToken
  messageId: string
  resolve: (value: any) => void
  reject: (reason?: any) => void

  constructor(
    message: any,
    sender: AgentId | null,
    recipient: AgentId,
    cancellationToken: CancellationToken,
    messageId: string,
    resolve: (value: any) => void,
    reject: (reason?: any) => void
  ) {
    this.message = message
    this.sender = sender
    this.recipient = recipient
    this.cancellationToken = cancellationToken
    this.messageId = messageId
    this.resolve = resolve
    this.reject = reject
  }
}

/**
 * A message envelope for sending a response to a message.
 */
@experimental
class ResponseMessageEnvelope {
  message: any
  sender: AgentId
  recipient: AgentId | null
  resolve: (value: any) => void

  constructor(message: any, sender: AgentId, recipient: AgentId | null, resolve: (value: any) => void) {
    this.message = message
    this.sender = sender
    this.recipient = recipient
    this.resolve = resolve
  }
}

type MessageEnvelope = PublishMessageEnvelope | SendMessageEnvelope | ResponseMessageEnvelope

/**
 * Subscription manager to handle topic subscriptions.
 */
class SubscriptionManager {
  private _subscriptions: Subscription[] = []
  private _seenTopics: Set<string> = new Set()
  private _subscribedRecipients: Map<string, AgentId[]> = new Map()

  get subscriptions(): Subscription[] {
    return this._subscriptions
  }

  async addSubscription(subscription: Subscription): Promise<void> {
    // Check if the subscription already exists
    if (this._subscriptions.some((sub) => sub.equals(subscription))) {
      throw new Error('Subscription already exists')
    }

    this._subscriptions.push(subscription)
    this._rebuildSubscriptions(this._seenTopics)
  }

  async removeSubscription(id: string): Promise<void> {
    // Check if the subscription exists
    if (!this._subscriptions.some((sub) => sub.id === id)) {
      throw new Error('Subscription does not exist')
    }

    this._subscriptions = this._subscriptions.filter((sub) => sub.id !== id)

    // Rebuild the subscriptions
    this._rebuildSubscriptions(this._seenTopics)
  }

  async getSubscribedRecipients(topicId: TopicId): Promise<AgentId[]> {
    const topicKey = topicId.toString()
    if (!this._seenTopics.has(topicKey)) {
      this._buildForNewTopic(topicId)
    }
    return this._subscribedRecipients.get(topicKey) || []
  }

  private _rebuildSubscriptions(topics: Set<string>): void {
    this._subscribedRecipients.clear()
    for (const topicKey of topics) {
      const topicId = TopicId.fromString(topicKey)
      this._buildForNewTopic(topicId)
    }
  }

  private _buildForNewTopic(topicId: TopicId): void {
    const topicKey = topicId.toString()
    this._seenTopics.add(topicKey)

    const recipients: AgentId[] = []
    for (const subscription of this._subscriptions) {
      if (subscription.isMatch(topicId)) {
        recipients.push(subscription.mapToAgent(topicId))
      }
    }
    this._subscribedRecipients.set(topicKey, recipients)
  }
}

/**
 * An in-process runtime that processes all messages using a single queue.
 * Messages are delivered in the order they are received.
 */
@experimental
export class InProcessRuntime implements CoreRuntime {
  public _messageQueue: MessageEnvelope[] = []
  public _shutdown = false
  public _processing = false

  private _agentFactories: Map<string, () => Agent | Promise<Agent>> = new Map()
  private _instantiatedAgents: Map<string, Agent> = new Map()
  private _backgroundTasks: Set<Promise<any>> = new Set()
  private _subscriptionManager = new SubscriptionManager()
  private _runContext: RunContext | null = null
  private _backgroundException: Error | null = null
  private _interventionHandlers: InterventionHandler[] | null = null
  private _serializationRegistry = new SerializationRegistry()

  constructor(options?: { interventionHandlers?: InterventionHandler[] }) {
    this._interventionHandlers = options?.interventionHandlers || null
  }

  /**
   * Get the number of unprocessed messages in the queue.
   */
  get unprocessedMessagesCount(): number {
    return this._messageQueue.length
  }

  /**
   * Send a message to an agent and get a response.
   */
  async sendMessage(
    message: any,
    recipient: AgentId,
    options?: {
      sender?: AgentId
      cancellationToken?: CancellationToken
      messageId?: string
    }
  ): Promise<any> {
    const cancellationToken = options?.cancellationToken || new CancellationToken()
    const messageId = options?.messageId || randomUUID()
    const sender = options?.sender || null

    const content = (message as any).__dict || message
    logger.info(`Sending message of type ${message.constructor.name} to ${recipient.type}: ${JSON.stringify(content)}`)

    logMessageEvent({
      payload: this._trySerialize(message),
      sender,
      receiver: recipient,
      kind: MessageKind.DIRECT,
      deliveryStage: DeliveryStage.SEND,
    })

    return new Promise((resolve, reject) => {
      if (!this._agentFactories.has(recipient.type)) {
        reject(new Error('Recipient not found'))
        return
      }

      const envelope = new SendMessageEnvelope(
        message,
        sender,
        recipient,
        cancellationToken,
        messageId,
        resolve,
        reject
      )
      this._messageQueue.push(envelope)
    })
  }

  /**
   * Publish a message to all agents that are subscribed to the topic.
   */
  async publishMessage(
    message: any,
    topicId: TopicId,
    options?: {
      sender?: AgentId
      cancellationToken?: CancellationToken
      messageId?: string
    }
  ): Promise<void> {
    const cancellationToken = options?.cancellationToken || new CancellationToken()
    const messageId = options?.messageId || randomUUID()
    const sender = options?.sender || null

    const content = (message as any).__dict || message
    logger.info(`Publishing message of type ${message.constructor.name} to all subscribers: ${JSON.stringify(content)}`)

    logMessageEvent({
      payload: this._trySerialize(message),
      sender,
      receiver: topicId,
      kind: MessageKind.PUBLISH,
      deliveryStage: DeliveryStage.SEND,
    })

    const envelope = new PublishMessageEnvelope(message, cancellationToken, sender, topicId, messageId)
    this._messageQueue.push(envelope)
  }

  /**
   * Process the next message in the queue.
   */
  async processNext(): Promise<void> {
    await this._processNext()
  }

  async _processNext(): Promise<void> {
    if (this._backgroundException) {
      const e = this._backgroundException
      this._backgroundException = null
      this._shutdown = true
      throw e
    }

    if (this._shutdown) {
      if (this._backgroundException) {
        const e = this._backgroundException
        this._backgroundException = null
        throw e
      }
      return
    }

    const messageEnvelope = this._messageQueue.shift()
    if (!messageEnvelope) {
      // No message to process, wait a bit
      await new Promise((resolve) => setTimeout(resolve, 10))
      return
    }

    this._processing = true

    try {
      if (messageEnvelope instanceof SendMessageEnvelope) {
        // Apply intervention handlers for send messages
        if (this._interventionHandlers) {
          for (const handler of this._interventionHandlers) {
            try {
              const messageContext = new MessageContext({
                sender: messageEnvelope.sender || undefined,
                topicId: undefined,
                isRpc: true,
                cancellationToken: messageEnvelope.cancellationToken,
                messageId: messageEnvelope.messageId,
              })
              const tempMessage = await handler.onSend(
                messageEnvelope.message,
                messageContext,
                messageEnvelope.recipient
              )
              warnIfUndefined(tempMessage, 'onSend')

              if (tempMessage === DropMessage) {
                logger.info(`Message dropped by intervention handler for recipient ${messageEnvelope.recipient.type}`)
                logMessageDroppedEvent({
                  payload: this._trySerialize(messageEnvelope.message),
                  sender: messageEnvelope.sender,
                  receiver: messageEnvelope.recipient,
                  kind: MessageKind.DIRECT,
                })
                messageEnvelope.reject(new MessageDroppedException())
                this._processing = false
                return
              }
              messageEnvelope.message = tempMessage
            } catch (error) {
              messageEnvelope.reject(error)
              this._processing = false
              return
            }
          }
        }

        const task = this._processSend(messageEnvelope)
        this._backgroundTasks.add(task)
        task.finally(() => this._backgroundTasks.delete(task))
      } else if (messageEnvelope instanceof PublishMessageEnvelope) {
        // Apply intervention handlers for publish messages
        if (this._interventionHandlers) {
          for (const handler of this._interventionHandlers) {
            try {
              const messageContext = new MessageContext({
                sender: messageEnvelope.sender || undefined,
                topicId: messageEnvelope.topicId,
                isRpc: false,
                cancellationToken: messageEnvelope.cancellationToken,
                messageId: messageEnvelope.messageId,
              })
              const tempMessage = await handler.onPublish(messageEnvelope.message, messageContext)
              warnIfUndefined(tempMessage, 'onPublish')

              if (tempMessage === DropMessage) {
                logger.info(
                  `Published message dropped by intervention handler for topic ${messageEnvelope.topicId.toString()}`
                )
                logMessageDroppedEvent({
                  payload: this._trySerialize(messageEnvelope.message),
                  sender: messageEnvelope.sender,
                  receiver: messageEnvelope.topicId,
                  kind: MessageKind.PUBLISH,
                })
                this._processing = false
                return
              }
              messageEnvelope.message = tempMessage
            } catch (error) {
              logger.error('Exception raised in intervention handler:', error)
              this._processing = false
              return
            }
          }
        }

        const task = this._processPublish(messageEnvelope)
        this._backgroundTasks.add(task)
        task.finally(() => this._backgroundTasks.delete(task))
      } else if (messageEnvelope instanceof ResponseMessageEnvelope) {
        // Apply intervention handlers for response messages
        if (this._interventionHandlers) {
          for (const handler of this._interventionHandlers) {
            try {
              const tempMessage = await handler.onResponse(
                messageEnvelope.message,
                messageEnvelope.sender,
                messageEnvelope.recipient
              )
              warnIfUndefined(tempMessage, 'onResponse')

              if (tempMessage === DropMessage) {
                logger.info(`Response message dropped by intervention handler from ${messageEnvelope.sender.type}`)
                logMessageDroppedEvent({
                  payload: this._trySerialize(messageEnvelope.message),
                  sender: messageEnvelope.sender,
                  receiver: messageEnvelope.recipient,
                  kind: MessageKind.RESPOND,
                })
                messageEnvelope.resolve(new MessageDroppedException())
                this._processing = false
                return
              }
              messageEnvelope.message = tempMessage
            } catch (error) {
              messageEnvelope.resolve(error)
              this._processing = false
              return
            }
          }
        }

        const task = this._processResponse(messageEnvelope)
        this._backgroundTasks.add(task)
        task.finally(() => this._backgroundTasks.delete(task))
      }
    } finally {
      this._processing = false
    }

    // Yield control to allow other tasks to run
    await new Promise((resolve) => setImmediate(resolve))
  }

  private async _processSend(messageEnvelope: SendMessageEnvelope): Promise<void> {
    const recipient = messageEnvelope.recipient

    if (!this._agentFactories.has(recipient.type)) {
      messageEnvelope.reject(new Error(`Agent type '${recipient.type}' does not exist.`))
      return
    }

    try {
      const senderIdStr = messageEnvelope.sender ? messageEnvelope.sender.toString() : 'Unknown'
      logger.info(
        `Calling message handler for ${recipient} with message type ${messageEnvelope.message.constructor.name} sent by ${senderIdStr}`
      )

      logMessageEvent({
        payload: this._trySerialize(messageEnvelope.message),
        sender: messageEnvelope.sender,
        receiver: recipient,
        kind: MessageKind.DIRECT,
        deliveryStage: DeliveryStage.DELIVER,
      })

      const recipientAgent = await this._getAgent(recipient)

      const messageContext = new MessageContext({
        sender: messageEnvelope.sender || undefined,
        topicId: undefined,
        isRpc: true,
        cancellationToken: messageEnvelope.cancellationToken,
        messageId: messageEnvelope.messageId,
      })

      const response = await recipientAgent.onMessage(messageEnvelope.message, messageContext)

      logMessageEvent({
        payload: this._trySerialize(response),
        sender: messageEnvelope.recipient,
        receiver: messageEnvelope.sender,
        kind: MessageKind.RESPOND,
        deliveryStage: DeliveryStage.SEND,
      })

      // Queue the response
      const responseEnvelope = new ResponseMessageEnvelope(
        response,
        messageEnvelope.recipient,
        messageEnvelope.sender,
        messageEnvelope.resolve
      )
      this._messageQueue.push(responseEnvelope)
    } catch (error) {
      logMessageHandlerExceptionEvent({
        payload: this._trySerialize(messageEnvelope.message),
        handlingAgent: recipient,
        exception: error as Error,
      })
      messageEnvelope.reject(error)
    }
  }

  private async _processPublish(messageEnvelope: PublishMessageEnvelope): Promise<void> {
    try {
      const recipients = await this._subscriptionManager.getSubscribedRecipients(messageEnvelope.topicId)
      const responses: Promise<any>[] = []

      for (const agentId of recipients) {
        // Avoid sending the message back to the sender
        if (messageEnvelope.sender && agentId.toString() === messageEnvelope.sender.toString()) {
          continue
        }

        const senderName = messageEnvelope.sender ? messageEnvelope.sender.toString() : 'Unknown'
        logger.info(
          `Calling message handler for ${agentId.type} with message type ${messageEnvelope.message.constructor.name} published by ${senderName}`
        )

        logMessageEvent({
          payload: this._trySerialize(messageEnvelope.message),
          sender: messageEnvelope.sender,
          receiver: null,
          kind: MessageKind.PUBLISH,
          deliveryStage: DeliveryStage.DELIVER,
        })

        const messageContext = new MessageContext({
          sender: messageEnvelope.sender || undefined,
          topicId: messageEnvelope.topicId,
          isRpc: false,
          cancellationToken: messageEnvelope.cancellationToken,
          messageId: messageEnvelope.messageId,
        })

        const agent = await this._getAgent(agentId)

        const onMessage = async (): Promise<any> => {
          try {
            return await agent.onMessage(messageEnvelope.message, messageContext)
          } catch (error) {
            logger.error(`Error processing publish message for ${agentId}`, error)
            logMessageHandlerExceptionEvent({
              payload: this._trySerialize(messageEnvelope.message),
              handlingAgent: agentId,
              exception: error as Error,
            })
            throw error
          }
        }

        responses.push(onMessage())
      }

      await Promise.all(responses)
    } catch (error) {
      this._backgroundException = error as Error
    }
  }

  private async _processResponse(messageEnvelope: ResponseMessageEnvelope): Promise<void> {
    const content = (messageEnvelope.message as any).__dict || messageEnvelope.message
    logger.info(
      `Resolving response with message type ${messageEnvelope.message.constructor.name} for recipient ${messageEnvelope.recipient} from ${messageEnvelope.sender.type}: ${JSON.stringify(content)}`
    )

    logMessageEvent({
      payload: this._trySerialize(content),
      sender: messageEnvelope.sender,
      receiver: messageEnvelope.recipient,
      kind: MessageKind.RESPOND,
      deliveryStage: DeliveryStage.DELIVER,
    })

    messageEnvelope.resolve(messageEnvelope.message)
  }

  /**
   * Start the runtime message processing loop.
   */
  start(): void {
    if (this._runContext) {
      throw new Error('Runtime is already started')
    }
    this._runContext = new RunContext(this)
  }

  /**
   * Close the runtime and all instantiated agents.
   */
  async close(): Promise<void> {
    if (this._runContext) {
      await this.stop()
    }

    // Close all the agents that have been instantiated
    for (const [_, agent] of this._instantiatedAgents) {
      await agent.close()
    }
  }

  /**
   * Immediately stop the runtime message processing loop.
   */
  async stop(): Promise<void> {
    if (!this._runContext) {
      throw new Error('Runtime is not started')
    }

    try {
      await this._runContext.stop()
    } finally {
      this._runContext = null
      this._messageQueue = []
    }
  }

  /**
   * Stop the runtime when there are no outstanding messages.
   */
  async stopWhenIdle(): Promise<void> {
    if (!this._runContext) {
      throw new Error('Runtime is not started')
    }

    try {
      await this._runContext.stopWhenIdle()
    } finally {
      this._runContext = null
      this._messageQueue = []
    }
  }

  /**
   * Stop the runtime when the condition is met.
   */
  async stopWhen(condition: () => boolean): Promise<void> {
    if (!this._runContext) {
      throw new Error('Runtime is not started')
    }

    await this._runContext.stopWhen(condition)
    this._runContext = null
    this._messageQueue = []
  }

  /**
   * Get the metadata for an agent.
   */
  async agentMetadata(agent: AgentId): Promise<AgentMetadata> {
    return (await this._getAgent(agent)).metadata
  }

  /**
   * Save the state of a single agent.
   */
  async agentSaveState(agent: AgentId): Promise<Record<string, any>> {
    return await (await this._getAgent(agent)).saveState()
  }

  /**
   * Load the state of a single agent.
   */
  async agentLoadState(agent: AgentId, state: Record<string, any>): Promise<void> {
    await (await this._getAgent(agent)).loadState(state)
  }

  /**
   * Register a factory for creating agents.
   */
  async registerFactory<T extends Agent>(
    type: string | AgentType,
    agentFactory: () => T | Promise<T>,
    options?: {
      expectedClass?: new (...args: any[]) => T
    }
  ): Promise<AgentType> {
    const agentType = typeof type === 'string' ? new CoreAgentType(type) : type

    if (this._agentFactories.has(agentType.type)) {
      throw new Error(`Agent with type ${agentType} already exists.`)
    }

    const factoryWrapper = async (): Promise<T> => {
      const agentInstance = await agentFactory()

      if (options?.expectedClass && !(agentInstance instanceof options.expectedClass)) {
        throw new Error('Factory registered using the wrong type.')
      }

      return agentInstance
    }

    this._agentFactories.set(agentType.type, factoryWrapper)

    return agentType
  }

  private async _getAgent(agentId: AgentId): Promise<Agent> {
    const key = agentId.toString()

    if (this._instantiatedAgents.has(key)) {
      return this._instantiatedAgents.get(key)!
    }

    if (!this._agentFactories.has(agentId.type)) {
      throw new Error(`Agent with name ${agentId.type} not found.`)
    }

    try {
      const agentFactory = this._agentFactories.get(agentId.type)!
      // Populate the context during agent instantiation so agents can access runtime and ID
      const agent = await AgentInstantiationContext.populateContext(this as any, agentId, () => agentFactory())
      this._instantiatedAgents.set(key, agent)
      return agent
    } catch (error) {
      logAgentConstructionExceptionEvent({
        agentId,
        exception: error as Error,
      })
      logger.error(`Error constructing agent ${agentId}`, error)
      throw error
    }
  }

  /**
   * Try to get the underlying agent instance.
   */
  async tryGetUnderlyingAgentInstance<T extends Agent>(id: AgentId, type?: new (...args: any[]) => T): Promise<T> {
    if (!this._agentFactories.has(id.type)) {
      throw new Error(`Agent with name ${id.type} not found.`)
    }

    const agentInstance = await this._getAgent(id)

    if (type && !(agentInstance instanceof type)) {
      throw new TypeError(
        `Agent with name ${id.type} is not of type ${type.name}. It is of type ${agentInstance.constructor.name}`
      )
    }

    return agentInstance as T
  }

  /**
   * Add a subscription to the runtime.
   */
  async addSubscription(subscription: Subscription): Promise<void> {
    await this._subscriptionManager.addSubscription(subscription)
  }

  /**
   * Remove a subscription from the runtime.
   */
  async removeSubscription(id: string): Promise<void> {
    await this._subscriptionManager.removeSubscription(id)
  }

  /**
   * Get an agent by id or type.
   */
  async get(
    idOrType: AgentId | AgentType | string,
    key: string = 'default',
    options?: { lazy?: boolean }
  ): Promise<AgentId> {
    const lazy = options?.lazy ?? true

    if (typeof idOrType === 'string') {
      const agentId = new CoreAgentId(idOrType, key)
      if (!lazy) {
        await this._getAgent(agentId)
      }
      return agentId
    } else if ('key' in idOrType) {
      // It's an AgentId (has both 'type' and 'key' properties)
      if (!lazy) {
        await this._getAgent(idOrType)
      }
      return idOrType
    } else {
      // It's an AgentType (has only 'type' property)
      const agentId = new CoreAgentId(idOrType.type, key)
      if (!lazy) {
        await this._getAgent(agentId)
      }
      return agentId
    }
  }

  /**
   * Save the state of all instantiated agents.
   */
  async saveState(): Promise<Record<string, any>> {
    const state: Record<string, Record<string, any>> = {}
    for (const [agentIdStr, _] of this._instantiatedAgents) {
      const agentId = CoreAgentId.fromString(agentIdStr)
      state[agentIdStr] = await (await this._getAgent(agentId)).saveState()
    }
    return state
  }

  /**
   * Load the state of all instantiated agents.
   */
  async loadState(state: Record<string, any>): Promise<void> {
    for (const agentIdStr in state) {
      const agentId = CoreAgentId.fromString(agentIdStr)
      if (this._agentFactories.has(agentId.type)) {
        await (await this._getAgent(agentId)).loadState(state[agentIdStr])
      }
    }
  }

  /**
   * Add a message serializer to the runtime.
   */
  addMessageSerializer(serializer: MessageSerializer<any> | MessageSerializer<any>[]): void {
    this._serializationRegistry.addSerializer(serializer)
  }

  /**
   * Try to serialize a message to a string representation.
   * Returns "Message could not be serialized" if serialization fails.
   */
  private _trySerialize(message: any): string {
    try {
      const typeName = this._serializationRegistry.getTypeName(message)
      const payload = this._serializationRegistry.serialize(message, {
        typeName,
        dataContentType: JSON_DATA_CONTENT_TYPE,
      })
      return new TextDecoder().decode(payload)
    } catch (error) {
      return 'Message could not be serialized'
    }
  }
}

/**
 * A context for the runtime to run in a background task.
 */
@experimental
class RunContext {
  private _runtime: InProcessRuntime
  private _runPromise: Promise<void> | null = null
  private _stopped = false

  constructor(runtime: InProcessRuntime) {
    this._runtime = runtime
    this._runPromise = this._run()
  }

  private async _run(): Promise<void> {
    while (!this._stopped) {
      await this._runtime._processNext()
    }
  }

  async stop(): Promise<void> {
    this._stopped = true
    this._runtime._shutdown = true
    if (this._runPromise) {
      await this._runPromise
    }
  }

  async stopWhenIdle(): Promise<void> {
    // Wait until the queue is empty
    while (this._runtime._messageQueue.length > 0 || this._runtime._processing) {
      await new Promise((resolve) => setTimeout(resolve, 10))
    }
    await this.stop()
  }

  async stopWhen(condition: () => boolean, checkPeriod: number = 1000): Promise<void> {
    const checkCondition = async (): Promise<void> => {
      while (!condition()) {
        await new Promise((resolve) => setTimeout(resolve, checkPeriod))
      }
      await this.stop()
    }
    await checkCondition()
  }
}
