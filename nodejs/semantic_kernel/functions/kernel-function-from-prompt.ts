import { PromptExecutionSettings } from '../connectors/ai/prompt-execution-settings'
import { FunctionInvocationContext } from '../filters/functions/function-invocation-context'
import { KERNEL_TEMPLATE_FORMAT_NAME, TemplateFormatTypes } from '../prompt-template/const'
import { PromptTemplateBase } from '../prompt-template/prompt-template-base'
import { PromptTemplateConfig } from '../prompt-template/prompt-template-config'
import { TEMPLATE_FORMAT_MAP } from '../prompt-template/template-format-map'
import { KernelFunction } from './kernel-function'
import { KernelFunctionMetadata } from './kernel-function-metadata'
import { KernelParameterMetadata } from './kernel-parameter-metadata'

const PROMPT_RETURN_PARAM: KernelParameterMetadata = new KernelParameterMetadata({
  name: 'return',
  description: 'The completion result',
  defaultValue: null,
  type: 'FunctionResult',
  isRequired: true,
})

/**
 * Options for creating a KernelFunctionFromPrompt.
 */
export interface KernelFunctionFromPromptOptions {
  functionName: string
  pluginName?: string
  description?: string
  prompt?: string
  templateFormat?: TemplateFormatTypes
  promptTemplate?: PromptTemplateBase
  promptTemplateConfig?: PromptTemplateConfig
  promptExecutionSettings?: PromptExecutionSettings | PromptExecutionSettings[] | Map<string, PromptExecutionSettings>
}

/**
 * Semantic Kernel Function from a prompt.
 */
export class KernelFunctionFromPrompt extends KernelFunction {
  /**
   * The prompt template.
   */
  public promptTemplate: PromptTemplateBase

  /**
   * The prompt execution settings.
   */
  public promptExecutionSettings: Map<string, PromptExecutionSettings>

  /**
   * Creates a new KernelFunctionFromPrompt instance.
   * @param options - The options for creating the function.
   */
  constructor(options: KernelFunctionFromPromptOptions) {
    const {
      functionName,
      pluginName,
      description,
      prompt,
      templateFormat = KERNEL_TEMPLATE_FORMAT_NAME,
      promptTemplate,
      promptTemplateConfig,
      promptExecutionSettings,
    } = options

    // Validate that we have a prompt source
    if (!prompt && !promptTemplateConfig && !promptTemplate) {
      throw new Error(
        'The prompt cannot be empty, must be supplied directly, through promptTemplateConfig or in the promptTemplate.'
      )
    }

    // Log warnings if conflicting parameters are provided
    if (prompt && promptTemplateConfig && promptTemplateConfig.template !== prompt) {
      console.warn(
        `Prompt (${prompt}) and PromptTemplateConfig (${promptTemplateConfig.template}) both supplied, ` +
          'using the template in PromptTemplateConfig, ignoring prompt.'
      )
    }
    if (templateFormat && promptTemplateConfig && promptTemplateConfig.templateFormat !== templateFormat) {
      console.warn(
        `Template (${templateFormat}) and PromptTemplateConfig (${promptTemplateConfig.templateFormat}) ` +
          'both supplied, using the template format in PromptTemplateConfig, ignoring template.'
      )
    }

    // Create prompt template if not provided
    let finalPromptTemplate = promptTemplate
    if (!finalPromptTemplate) {
      let finalConfig = promptTemplateConfig
      if (!finalConfig) {
        // Create config from prompt
        finalConfig = new PromptTemplateConfig({
          name: functionName,
          description,
          template: prompt!,
          templateFormat,
        })
      } else if (!finalConfig.template) {
        finalConfig.template = prompt!
      }

      // Create template from format map
      const TemplateClass = TEMPLATE_FORMAT_MAP[finalConfig.templateFormat]
      if (!TemplateClass) {
        throw new Error(`Unknown template format: ${finalConfig.templateFormat}`)
      }
      finalPromptTemplate = new TemplateClass(finalConfig)
    }

    // Create metadata
    let metadata: KernelFunctionMetadata
    try {
      metadata = new KernelFunctionMetadata({
        name: functionName,
        pluginName,
        description: description || '',
        parameters: finalPromptTemplate.promptTemplateConfig.getKernelParameterMetadata(),
        isPrompt: true,
        isAsynchronous: true,
        returnParameter: PROMPT_RETURN_PARAM,
      })
    } catch (exc) {
      throw new Error(`Failed to create KernelFunctionMetadata: ${exc}`)
    }

    super(metadata)

    this.promptTemplate = finalPromptTemplate

    // Handle prompt execution settings
    this.promptExecutionSettings = new Map()

    // Use execution settings from promptTemplateConfig if not explicitly provided
    let finalExecutionSettings = promptExecutionSettings
    if (!finalExecutionSettings && finalPromptTemplate.promptTemplateConfig.executionSettings) {
      finalExecutionSettings = finalPromptTemplate.promptTemplateConfig.executionSettings
    }

    if (finalExecutionSettings) {
      if (finalExecutionSettings instanceof Map) {
        this.promptExecutionSettings = finalExecutionSettings
      } else if (Array.isArray(finalExecutionSettings)) {
        for (const setting of finalExecutionSettings) {
          const serviceName = (setting as any).service_id || setting.serviceId || 'default'
          this.promptExecutionSettings.set(serviceName, setting)
        }
      } else {
        const serviceName = (finalExecutionSettings as any).service_id || finalExecutionSettings.serviceId || 'default'
        this.promptExecutionSettings.set(serviceName, finalExecutionSettings)
      }
    }
  }

  /**
   * Internal invoke method implementation.
   * @param context - The function invocation context.
   */
  protected async _invokeInternal(context: FunctionInvocationContext): Promise<void> {
    // Render the prompt
    const renderedPrompt = await this.promptTemplate.render(context.kernel, context.arguments)

    // Get AI service and invoke
    // This is a simplified implementation - full implementation would:
    // 1. Get the appropriate AI service from kernel
    // 2. Apply prompt execution settings
    // 3. Invoke the service
    // 4. Handle filters and hooks
    // For now, we'll throw an error indicating incomplete implementation
    throw new Error(
      `KernelFunctionFromPrompt._invokeInternal not fully implemented yet. Rendered prompt: ${renderedPrompt.substring(0, 100)}...`
    )
  }

  /**
   * Internal streaming invoke method implementation.
   * @param context - The function invocation context.
   */
  protected async _invokeInternalStream(context: FunctionInvocationContext): Promise<void> {
    // Render the prompt
    const renderedPrompt = await this.promptTemplate.render(context.kernel, context.arguments)

    // This would stream results from the AI service
    throw new Error(
      `KernelFunctionFromPrompt._invokeInternalStream not fully implemented yet. Rendered prompt: ${renderedPrompt.substring(0, 100)}...`
    )
  }
}
