export type TranscriptionRequest = {
  file: Blob;
  /** Override the file name, or supply one for an unnamed Blob. */
  filename?: string;
  /** ISO-639-1 language code, for example "de". Omit for automatic detection. */
  language?: string;
  /** Optional vocabulary or context hint, subject to model support. */
  prompt?: string;
  signal?: AbortSignal;
};

export type TranscriptionResult = {
  text: string;
};

export type TranscriptionProvider = {
  name: string;
  model: string;
  transcribe(request: TranscriptionRequest): Promise<TranscriptionResult>;
};
