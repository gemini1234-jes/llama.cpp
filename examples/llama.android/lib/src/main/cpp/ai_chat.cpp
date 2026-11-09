
extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    // Reset short-term states
    reset_short_term_states();

    // Obtain user prompt
    const auto *const user_prompt =
            env->GetStringUTFChars(juser_prompt, nullptr);

    if (user_prompt == nullptr) {
        LOGe("%s: Failed to obtain user prompt", __func__);
        return 1;
    }

    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);

    std::string formatted_user_prompt(user_prompt);

    // Format user prompt if chat template is available
    const bool has_chat_template =
            common_chat_templates_was_explicit(g_chat_templates.get());

    if (has_chat_template) {
        formatted_user_prompt =
                chat_add_and_format(ROLE_USER, user_prompt);
    }

    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    // Tokenize user prompt
    auto user_tokens = common_tokenize(
            g_context,
            formatted_user_prompt,
            has_chat_template,
            has_chat_template
    );

    for (auto id : user_tokens) {
        LOGv("token: `%s`\t -> `%d`",
             common_token_to_piece(g_context, id).c_str(), id);
    }

    // Check context size
    const int max_batch_size =
            DEFAULT_CONTEXT_SIZE - OVERFLOW_HEADROOM;

    if ((int) user_tokens.size() > max_batch_size) {
        const int skipped_tokens =
                (int) user_tokens.size() - max_batch_size;

        user_tokens.resize(max_batch_size);

        LOGw("%s: User prompt too long! Skipped %d tokens!",
             __func__, skipped_tokens);
    }

    // Use the actual token count after truncation
    const int user_prompt_size = (int) user_tokens.size();

    if (user_prompt_size == 0) {
        LOGe("%s: User prompt contains no tokens", __func__);
        return 1;
    }

    // Decode user tokens
    if (decode_tokens_in_batches(
            g_context,
            g_batch,
            user_tokens,
            current_position,
            true
    )) {
        LOGe("%s: llama_decode() failed!", __func__);
        return 2;
    }

    // Update current position
    current_position += user_prompt_size;

    // Set correct generation stopping position
    stop_generation_position =
            current_position + std::max(0, (int)n_predict);

    LOGi("%s: User prompt processed. Current position: %d, Stop position: %d",
         __func__, current_position, stop_generation_position);

    return 0;
}
