// Stub for OrtSessionOptionsAppendExecutionProvider_Nnapi.
// sherpa-onnx's session.cc references this symbol but the ONNX Runtime
// static library doesn't include the NNAPI provider.

typedef struct OrtStatus OrtStatus;
typedef struct OrtSessionOptions OrtSessionOptions;

OrtStatus* OrtSessionOptionsAppendExecutionProvider_Nnapi(
    OrtSessionOptions* options, unsigned int nnapi_flags) {
    (void)options;
    (void)nnapi_flags;
    return (OrtStatus*)0;
}
