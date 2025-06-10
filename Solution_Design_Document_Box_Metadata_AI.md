# Solution Design Document: Box Metadata AI Application

## 1. Introduction/Overview

### 1.1. Purpose of the Document
This document outlines the solution design for the Box Metadata AI Application. It details the system architecture, components, data flows, and key logic areas to guide development, testing, and future enhancements.

### 1.2. Project Objectives
The primary objective of this project is to develop an AI-powered application that integrates with Box to automatically categorize documents and extract relevant metadata. This includes:
*   Securely authenticating users via Box OAuth2.
*   Allowing users to select files from their Box account.
*   Categorizing selected documents using AI models.
*   Extracting structured and freeform metadata from documents based on their category.
*   Providing confidence scores for both categorization and metadata extraction.
*   Enabling users to review, edit, and approve the AI-generated metadata.
*   Applying the approved metadata back to the files in Box.
*   Supporting various AI prompting strategies and consensus mechanisms for improved accuracy.
*   Implementing robust validation and error handling.

### 1.3. Intended Audience
This document is intended for:
*   Software Developers and Engineers
*   System Architects
*   Quality Assurance and Testers
*   Project Managers
*   Product Owners
*   Technical Support Personnel

## 2. High-Level Design

### 2.1. System Architecture Overview
The application is a Streamlit web application that interacts with the Box API for file management and metadata operations. The core logic is implemented in Python and leverages AI models (presumably via API calls to services like OpenAI, Azure OpenAI, or similar) for document categorization and metadata extraction.

The key components are:
*   **Streamlit Frontend:** Provides the user interface for authentication, file selection, configuration, results display, and user interaction.
*   **Python Backend:** Handles the core application logic, including:
    *   Box API communication (authentication, file operations, metadata application).
    *   AI model interaction (prompt generation, API calls, response parsing).
    *   Document categorization logic.
    *   Metadata extraction workflows.
    *   Confidence scoring and adjustment.
    *   Validation of extracted metadata.
    *   Orchestration of the processing pipeline.
*   **Box Platform:** Serves as the repository for documents and their associated metadata.
*   **AI Service(s):** External AI platforms that perform the natural language processing tasks.

### 2.2. Process Flow (User Journey)
1.  **Authentication:**
    *   User accesses the Streamlit application.
    *   User is redirected to Box for OAuth2 authentication if not already authenticated.
    *   Upon successful authentication, the application receives an access token.
2.  **File Selection & Configuration:**
    *   User selects one or more files/folders from their Box account using a file picker.
    *   User configures processing options:
        *   Categorization mode (e.g., Standard, Detailed, Sequential Consensus).
        *   Extraction type (Structured, Freeform).
        *   Metadata templates/schemas.
        *   Validation rules.
3.  **Document Categorization (if applicable):**
    *   The application sends document content (or summaries) to an AI model based on the selected categorization strategy.
    *   AI returns a document category and an initial confidence score.
    *   The system calculates a final confidence score for categorization, potentially involving multi-factor analysis or consensus mechanisms.
4.  **Metadata Extraction:**
    *   Based on the document category (or if categorization is skipped), the appropriate metadata template/schema is selected.
    *   For **Structured Extraction**:
        *   Field definitions are prepared.
        *   A detailed prompt is constructed, including field definitions and document content, requesting JSON output with values and field-level confidences.
        *   The AI model processes the request and returns structured metadata.
    *   For **Freeform Extraction**:
        *   A prompt is constructed to extract relevant information without a predefined schema, requesting JSON output with identified entities and confidences.
    *   The AI's response is parsed.
5.  **Validation and Confidence Adjustment (for Metadata):**
    *   Extracted metadata is validated against predefined rules (e.g., data types, mandatory fields).
    *   Confidence scores for metadata fields are adjusted based on AI self-reported confidence, validation results, and other factors (e.g., penalties for missing mandatory fields).
    *   An overall document status (e.g., "High Confidence," "Needs Review") is determined.
6.  **Review and Editing:**
    *   The application displays the categorized document type (if applicable) and extracted metadata along with their confidence scores.
    *   Users can review the AI's suggestions.
    *   Users can edit, add, or delete metadata fields and values.
7.  **Metadata Application:**
    *   User approves the (potentially edited) metadata.
    *   The application uses the Box API to apply the metadata to the corresponding files in Box, using the selected metadata template instance.
8.  **Reporting/Completion:**
    *   The application indicates the success or failure of the metadata application process for each file.
    *   Batch processing results are summarized.

## 3. Low-Level Design / Component Design

### 3.1. AI Prompting Strategy

#### 3.1.1. Document Categorization
Prompts are dynamically generated based on the selected categorization mode.
*   **Standard Categorization:**
    *   *Prompt Focus:* Identify the single most relevant category for the document from a provided list.
    *   *Dynamic Content:* Document text (or a significant summary), list of possible categories.
    *   *Example Snippet (Conceptual):*
        ```
        System: You are a document categorization assistant. Based on the following text, classify the document into one of these categories: {{categories_list}}. Provide only the category name and a confidence score (0.0-1.0).

        User: {{document_text}}
        ```
*   **Detailed Categorization:**
    *   *Prompt Focus:* Identify the most relevant category and potentially sub-categories or provide a rationale.
    *   *Dynamic Content:* Document text, detailed category definitions or hierarchy.
    *   *Example Snippet (Conceptual):*
        ```
        System: You are an expert document classifier. Analyze the following text and determine the primary category and any relevant sub-categories from the provided schema: {{category_schema}}. Explain your reasoning and provide a confidence score (0.0-1.0) for the primary category.

        User: {{document_text}}
        ```
*   **Sequential Consensus Categorization:**
    *   **Stage 1: Independent Analysis:**
        *   *Prompt Focus:* Similar to Standard Categorization, but for multiple AI "experts" working independently.
        *   *Dynamic Content:* Document text, list of categories.
        *   *Output Expectation:* Category and confidence.
    *   **Stage 2: Review & Critique (if disagreement in Stage 1):**
        *   *Prompt Focus:* Review previous categorizations and rationales from other "experts," identify discrepancies, and provide a revised categorization with justification.
        *   *Dynamic Content:* Document text, categories, outputs from Stage 1 (categories, confidences, rationales if any).
        *   *Example Snippet (Conceptual):*
            ```
            System: You are an AI review expert. Two previous AI assistants analyzed this document.
            AI_1 categorized it as '{{category_AI1}}' with confidence {{confidence_AI1}}.
            AI_2 categorized it as '{{category_AI2}}' with confidence {{confidence_AI2}}.
            Review their assessments and the document text. Provide your own refined category, confidence score (0.0-1.0), and a brief justification for your decision, especially if it differs.

            User: {{document_text}}
            ```
    *   **Stage 3: Arbitration (if disagreement persists after Stage 2):**
        *   *Prompt Focus:* Act as a final arbiter. Review all previous analyses and make a definitive categorization.
        *   *Dynamic Content:* Document text, categories, outputs from Stage 1 and Stage 2.
        *   *Example Snippet (Conceptual):*
            ```
            System: You are the final AI arbitrator. Multiple AI assistants have analyzed this document with differing conclusions.
            {{summary_of_previous_stages}}
            Review all information and the document text. Provide the definitive category and a confidence score (0.0-1.0).

            User: {{document_text}}
            ```

#### 3.1.2. Metadata Extraction
*   **Structured Extraction (e.g., using `extract_structured_metadata`):**
    *   *System Message Focus:* Instruct the AI to act as a metadata extraction specialist, analyze the document against a provided schema, and return a JSON object. Emphasize adherence to field types, required fields, and providing field-level confidence.
    *   *Dynamic Content:* Document text, JSON schema of metadata fields (field names, descriptions, types, options for dropdowns/multi-selects).
    *   *Prompt/System Message Example (Conceptual):*
        ```
        System: You are an AI assistant specialized in extracting metadata from documents.
        The user will provide document text and a metadata schema.
        Your task is to extract the relevant information from the text and structure it according to the schema.
        The output MUST be a single JSON object.
        For each field in the schema, provide its extracted value.
        If a field's value is not found, use `null` for that field.
        For each field, also provide a confidence level for your extraction: "High", "Medium", or "Low".

        Schema:
        {{json_schema_definition}}

        Respond only with the JSON object.

        User: {{document_text}}
        ```
    *   *Output Expectation:* A JSON object where keys are metadata field IDs, and each value is an object containing `value` and `confidence` (e.g., `{"fieldName": {"value": "extracted_text", "confidence": "High"}}`).
*   **Freeform Extraction:**
    *   *System Message Focus:* Instruct the AI to identify and extract key entities, topics, or summaries from the document without a rigid schema. Request JSON output for easier parsing.
    *   *Dynamic Content:* Document text, potentially a list of desired entity types or areas of interest.
    *   *Prompt/System Message Example (Conceptual):*
        ```
        System: You are an AI assistant for freeform information extraction.
        Analyze the provided document text and identify key entities, topics, and a brief summary.
        Structure your output as a JSON object with keys like "entities", "topics", "summary".
        For entities, list them and provide a confidence level ("High", "Medium", "Low") for each.

        User: {{document_text}}
        ```
    *   *Output Expectation:* A JSON object with flexible keys representing the extracted information (e.g., `{"summary": "...", "keywords": [{"word": "...", "confidence": "Medium"}]}`).

### 3.2. Confidence Scoring Mechanism

#### 3.2.1. Document Categorization
The final confidence score for document categorization is derived through a pipeline:
1.  **Initial AI Score Parsing:** The raw confidence score (typically 0.0-1.0 or High/Medium/Low) provided by the AI model for its chosen category is parsed. If qualitative, it's converted to a numeric scale (e.g., High=0.9, Medium=0.7, Low=0.4).
2.  **Consensus Mode Derivation (Pre-`calculate_multi_factor_confidence`):**
    *   **Parallel Consensus:** If multiple independent AI runs are performed (not sequential), their categorizations and confidences are compared. If a majority agrees on a category, that category is chosen. The confidence might be an average of the agreeing AIs, or the highest among them. If no consensus, the category with the highest individual confidence might be chosen, or it's flagged for review.
    *   **Sequential Consensus:**
        *   *Stage 1 (Independent):* Each "expert" provides a category and confidence.
        *   *Stage 2 (Review):* The reviewer provides a revised category and confidence. This becomes the current "best" guess.
        *   *Stage 3 (Arbitration):* The arbitrator provides the final category and confidence. This is the output used for subsequent steps.
    The output from these consensus modes (the chosen category and its associated raw/derived confidence) then feeds into `calculate_multi_factor_confidence`.
3.  **`calculate_multi_factor_confidence` (Conceptual Function):** This function adjusts the initial/consensus AI score based on various factors.
    *   *Factors (Examples):*
        *   **Keyword Presence:** Presence of specific keywords associated with certain categories might boost confidence.
        *   **Document Length/Structure:** Very short documents might have lower reliability.
        *   **Source Reliability (if known):** Not typically applicable here but a general factor.
        *   **Disagreement in Consensus:** High disagreement in early consensus stages might penalize the final confidence even if an arbitrator makes a choice.
    *   *Weights:* Factors are assigned weights based on their perceived importance in determining categorization accuracy.
    *   *Calculation:* A weighted sum or a more complex formula combines these factors with the initial AI score.
4.  **`apply_confidence_calibration` (Conceptual Function):** This function further refines the score from `calculate_multi_factor_confidence`.
    *   *Purpose:* To align AI confidence scores with observed accuracy, potentially using historical performance data or predefined calibration curves.
    *   *Method:* May involve scaling, binning, or applying a calibration model.
The result is the final, adjusted confidence score for the document's category.

#### 3.2.2. Metadata Extraction
The confidence for individual metadata fields and the overall document status are determined as follows:
1.  **AI Self-Reported Confidence:** The AI is prompted to provide a confidence level (e.g., "High", "Medium", "Low") for each extracted field value.
2.  **Numeric Conversion (`ConfidenceAdjuster` or similar logic):**
    *   Qualitative AI confidence (High/Medium/Low) is converted to a numeric score (e.g., High: 0.9, Medium: 0.7, Low: 0.4). This is the **Initial Numeric Confidence**.
3.  **Penalty Application (`ConfidenceAdjuster` or similar logic):** Penalties are applied to the Initial Numeric Confidence to calculate an **Adjusted Numeric Confidence**.
    *   **Validation Penalty:** If a field fails data type validation (e.g., text in a date field), its confidence is significantly reduced, possibly to a very low fixed value (e.g., 0.1) or by a large percentage.
    *   **Mandatory Field Penalty:** If a field is marked as mandatory but no value is extracted (or extracted value is null/empty), its confidence might be set to 0 or a very low score. If a value *is* extracted for a mandatory field but has low AI confidence, it might receive an additional penalty.
    *   **Low Initial AI Confidence Penalty:** If the AI itself reports "Low" confidence, this already results in a lower Initial Numeric Confidence. Further specific penalties might be applied if this is deemed particularly unreliable for certain fields.
4.  **Final Adjusted Numeric/Qualitative Scores:**
    *   The **Adjusted Numeric Confidence** (0.0-1.0) is stored for each field.
    *   This numeric score is often mapped back to a qualitative score (e.g., "High," "Medium," "Low," "Validation Error") for display purposes, using predefined thresholds (e.g., >0.85 = High, 0.6-0.85 = Medium, <0.6 = Low).
5.  **Overall Document Status (`get_overall_document_status` or similar logic):**
    *   This status reflects the aggregate confidence across all extracted fields for a document.
    *   *Logic (Example):*
        *   If any field has "Validation Error" -> "Needs Review (Validation Error)".
        *   If any mandatory field is missing or has "Low" adjusted confidence -> "Needs Review (Mandatory Missing/Low Confidence)".
        *   If a high percentage of fields have "Low" adjusted confidence -> "Needs Review (Generally Low Confidence)".
        *   If all fields have "High" adjusted confidence -> "High Confidence".
        *   Otherwise -> "Medium Confidence" or a similar intermediate status.
    *   This function iterates through the adjusted confidences of all relevant fields to make its determination.

### 3.3. Metadata Extraction Workflow (Core AI Interaction for Structured Data)

1.  **Template Selection (`get_metadata_template_id`):**
    *   This function determines which Box metadata template (schema) to use for extraction.
    *   It typically relies on mappings defined by the user or administrator: `document_category -> template_display_name_or_id`.
    *   If a document's category (from the categorization step) matches a key in this mapping, the corresponding template ID is retrieved.
    *   A fallback mechanism is essential: if no specific mapping exists for a category, a default template might be used, or extraction might be skipped/flagged.
2.  **Field Preparation (`get_fields_for_ai_from_template`):**
    *   Once a template ID is known, this function fetches the schema of that Box metadata template (likely using a Box SDK call like `client.get_metadata_template_by_id(template_id)` or `client.get_metadata_template_schema(template_key, scope)`).
    *   It parses the template schema (which is usually in JSON format) to extract definitions for each field.
    *   This includes the field's `key` (ID), `displayName`, `type` (string, number, date, enum), and `options` (for enum/dropdown fields).
    *   This information is formatted into a simplified JSON schema or textual description suitable for inclusion in the AI prompt, as shown in the Structured Extraction prompt example (Section 3.1.2).
3.  **AI API Call (`extract_structured_metadata`):**
    *   This function orchestrates the call to the AI model.
    *   It takes the document content (or relevant chunks) and the prepared field definitions (from `get_fields_for_ai_from_template`) as input.
    *   It constructs the full prompt, including the system message, the field definitions, and the document text.
    *   It makes an API call to the configured AI service (e.g., OpenAI `chat.completions.create`).
    *   It handles potential API errors (e.g., rate limits, timeouts, authentication issues).
4.  **Response Parsing:**
    *   The AI is expected to return a JSON string. This function parses the JSON string into a Python dictionary.
    *   It iterates through the keys (field IDs) in the AI's JSON response.
    *   For each field, it extracts the `value` and the AI-reported `confidence` (e.g., "High", "Medium", "Low").
    *   This parsed data (value and initial confidence for each field) is then passed to the Validation Engine and ConfidenceAdjuster.
    *   Robust error handling is crucial here to manage cases where the AI's output is not valid JSON or doesn't conform to the expected structure.

### 3.4. Validation Engine (`modules/validation_engine.py`)

*   **`ValidationRuleLoader`:**
    *   Responsible for loading validation rules, typically from an external configuration file (e.g., `validation_rules.json`).
    *   These rules define constraints for specific metadata fields, such as data type (e.g., date, integer, email), format (e.g., regex patterns for specific ID formats), value ranges, or allowed values for non-enum fields.
    *   Rules might be associated with specific metadata templates or apply globally.
*   **`Validator`:**
    *   This class or module applies the loaded validation rules to the extracted metadata.
    *   **Field-level Checks:** For each extracted field value, it checks against applicable rules (e.g., `is_date()`, `matches_regex(pattern)`).
    *   **Mandatory Field Checks:** It verifies if all fields marked as 'required' in the metadata template (or validation rules) have been assigned a non-null value.
    *   The output of the `Validator` is typically a list of validation errors or a status per field indicating whether it passed or failed validation, and the reason for failure.
*   **Role of `ConfidenceAdjuster` in Using Validation Outputs:**
    *   The `ConfidenceAdjuster` (as described in 3.2.2) consumes the validation results from the `Validator`.
    *   If a field fails validation, the `ConfidenceAdjuster` significantly penalizes its confidence score, often overriding the AI's self-reported confidence. This ensures that demonstrably incorrect data is flagged with very low confidence.

### 3.5. Orchestration (`modules/processing.py`)
The `processing.py` module likely contains the main orchestration logic that ties together the various stages of the document processing pipeline.
*   It manages the overall workflow for a single document or a batch of documents.
*   This includes:
    *   Initiating document categorization (if enabled).
    *   Selecting the appropriate metadata template based on the category.
    *   Calling `get_fields_for_ai_from_template` to prepare field definitions.
    *   Invoking `extract_structured_metadata` (or its freeform equivalent) to get AI extractions.
    *   Passing the results to the `ValidationEngine`.
    *   Using the `ConfidenceAdjuster` to calculate final field confidences.
    *   Determining the `get_overall_document_status`.
    *   Storing and managing the results for display in the UI.
    *   Handling exceptions and errors that occur during any stage of processing.
    *   Interfacing with the Streamlit session state to update progress and display results.

### 3.6. Batch Processing Overview (`modules/batch_utils.py`, `document_categorization_updated.py`)

*   **`batch_utils.py`:**
    *   This module likely provides utilities for managing batches of documents.
    *   This could include functions for:
        *   Dividing a large list of selected files into smaller, manageable batches.
        *   Iterating over batches and orchestrating their processing.
        *   Aggregating results from multiple batches.
        *   Handling rate limits or other constraints when dealing with many API calls in succession.
        *   Providing progress updates for batch operations.
*   **`document_categorization_updated.py` (or similar for batch categorization):**
    *   When document categorization is performed in batch mode, this module (or relevant functions within `processing.py` that use `batch_utils.py`) would handle sending multiple documents for categorization.
    *   It might optimize by preparing a batch of requests to the AI service if the service supports batch inputs.
    *   It collects categorization results for all documents in the batch before proceeding to the metadata extraction phase for each document.
    *   Similarly, batching can apply to metadata extraction if the AI service and workflow are designed to handle it, allowing multiple documents (with their respective schemas, if different) to be processed more efficiently.

## 4. Data Flow

Key data, particularly user session information and processing results, is managed using Streamlit's session state.
*   **`st.session_state.box_client`**: Stores the authenticated Box API client instance.
*   **`st.session_state.user_id`**: Stores the ID of the authenticated Box user.
*   **`st.session_state.selected_files`**: A list of Box file/folder objects selected by the user.
*   **`st.session_state.metadata_config`**: A dictionary or custom object holding user configurations like:
    *   `categorization_mode` (e.g., "Standard", "Detailed", "SequentialConsensus")
    *   `extraction_type` (e.g., "Structured", "Freeform")
    *   `selected_template_key_scope` (for structured extraction)
    *   `target_folder_id` (for saving processed files, if applicable)
*   **`st.session_state.document_categorization`**: A dictionary mapping file IDs to their categorization results:
    *   `{file_id: {"category": "...", "confidence": 0.85, "status": "Categorized"}}`
*   **`st.session_state.extraction_results`**: A dictionary mapping file IDs to their metadata extraction results:
    *   `{file_id: {"template_id": "...", "fields": {field_key: {"value": "...", "initial_confidence": "High", "adjusted_confidence_numeric": 0.9, "adjusted_confidence_qualitative": "High", "validation_status": "Pass"}}, "overall_status": "High Confidence"}}`
*   **`st.session_state.ai_prompts_log` (optional but good practice):** A list or dictionary storing prompts sent to the AI and raw responses for debugging and transparency.
*   **`st.session_state.validation_rules`**: Loaded validation rules from `validation_rules.json`.

**General Flow:**
1.  User authenticates -> `box_client`, `user_id` populated.
2.  User selects files -> `selected_files` populated.
3.  User sets configurations -> `metadata_config` populated.
4.  Processing starts:
    *   Categorization populates/updates `document_categorization`.
    *   Extraction uses `metadata_config` and `document_categorization` (for template selection), then populates/updates `extraction_results`.
5.  UI reads from `document_categorization` and `extraction_results` to display information.
6.  User edits are reflected in `extraction_results`.
7.  Upon applying metadata, `extraction_results` are used to make Box API calls.

## 5. Code Review Summary (New Section)

This section would typically be filled after a manual code review. Assuming a well-structured project based on the prompt:

*   **Overall Code Organization, Clarity, Modularity:**
    *   The codebase is expected to be organized into modules with specific responsibilities (e.g., `box_integration`, `ai_interface`, `validation_engine`, `processing`, `ui_components`).
    *   Clarity is generally good, with function and variable names being descriptive.
    *   Modularity seems to be a design goal, allowing components like the AI prompting strategy or validation rules to be updated or replaced with manageable impact.
*   **Error Handling Patterns:**
    *   Error handling is likely present for API calls (Box, AI) using try-except blocks.
    *   Custom exceptions might be defined for application-specific errors (e.g., `TemplateNotFoundError`, `AIResponseParseError`).
    *   User-facing errors are hopefully translated into friendly messages in the Streamlit UI.
    *   Robust handling of unexpected AI responses (e.g., malformed JSON, missing confidence scores) is critical and needs verification.
*   **Logging Practices:**
    *   A logging mechanism (e.g., Python's `logging` module) is expected to be in place.
    *   Key events, decisions, errors, and API interactions should be logged for debugging and auditing.
    *   Log levels (INFO, DEBUG, ERROR) should be used appropriately.
    *   Consider logging AI prompts and raw responses (selectively, respecting data privacy) for easier troubleshooting of AI behavior.
*   **Use of Streamlit Session State:**
    *   Session state is used extensively to maintain user context, selected files, configurations, and processing results across Streamlit's script reruns.
    *   Care should be taken to initialize session state variables correctly at the beginning of a session.
    *   Large objects in session state should be managed efficiently to avoid performance issues, though for typical metadata, this is less of a concern than, for example, large dataframes.
*   **Configuration Management:**
    *   Configurations like AI API keys, Box client ID/secret, logging levels, and application behavior parameters are likely managed via environment variables or external configuration files (e.g., `.env`, `config.yaml`).
    *   Validation rules are externalized (e.g., `validation_rules.json`), which is good practice for maintainability.
    *   Metadata template mappings (category to template ID) might also be in a configuration file or managed within the application settings UI.

## 6. Key Logic Areas (New Section - Summary for easy reference)

*   **AI Prompting for Categorization:**
    *   Dynamic prompt generation based on mode (Standard, Detailed, Sequential Consensus).
    *   Insertion of document content and category lists/schemas into prompts.
    *   Specific instructions for AI to return category and confidence.
    *   Multi-stage prompting for Sequential Consensus (Independent, Review, Arbitration).
*   **AI Prompting for Metadata Extraction:**
    *   System messages defining AI role and output format (JSON).
    *   **Structured:** Inclusion of field definitions (name, type, options) from Box schemas in the prompt. Request for field-level confidence.
    *   **Freeform:** More open-ended prompts, still requesting JSON and confidence where applicable.
*   **Confidence Calculation for Document Categorization:**
    *   Parsing initial AI score.
    *   Deriving a score from Parallel or Sequential Consensus methods.
    *   Potential use of `calculate_multi_factor_confidence` with weighted factors.
    *   Refinement via `apply_confidence_calibration`.
*   **Confidence Calculation for Metadata Extraction:**
    *   Conversion of AI's qualitative confidence (High/Medium/Low) to numeric.
    *   Application of penalties by `ConfidenceAdjuster` based on:
        *   Validation rule failures (e.g., data type mismatch).
        *   Missing mandatory fields.
        *   Low initial AI confidence.
    *   Calculation of adjusted numeric and qualitative confidence per field.
    *   Derivation of `get_overall_document_status` based on aggregated field confidences and validation status.
```
