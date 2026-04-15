#include "perfetto_singleton.hpp"

PerfettoSigleton PerfettoSigleton::instance;

// Reserves internal static storage for our tracing categories.
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

void PerfettoSigleton::InitializePerfetto() {
  perfetto::TracingInitArgs args;
  // The backends determine where trace events are recorded. For this example
  // we are going to use the in-process tracing service, which only includes
  // in-app events.
  args.backends = perfetto::kInProcessBackend;
  args.backends |= perfetto::kSystemBackend;
  perfetto::Tracing::Initialize(args);
  perfetto::TrackEvent::Register();
  // Enable the console interceptor.
  // perfetto::ConsoleInterceptor::Register();
  PERFETTO_LOG("System tracing initialized.");
}

std::unique_ptr<perfetto::TracingSession>
PerfettoSigleton::StartInprocessTracing() {
  // The trace config defines which types of data sources are enabled for
  // recording. In this example we just need the "track_event" data source,
  // which corresponds to the TRACE_EVENT trace points.
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(1024);
  auto *ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");
  // Enable the console interceptor.
  // ds_cfg->mutable_interceptor_config()->set_name("console");

  auto tracing_session = perfetto::Tracing::NewTrace();
  tracing_session->Setup(cfg);
  tracing_session->StartBlocking();
  return tracing_session;
}

void PerfettoSigleton::StopInprocessTracing(
    std::unique_ptr<perfetto::TracingSession> tracing_session) {
  // Make sure the last event is closed for this example.
  perfetto::TrackEvent::Flush();

  // Stop tracing and read the trace data.
  tracing_session->StopBlocking();
  std::vector<char> trace_data(tracing_session->ReadTraceBlocking());

  // Write the result into a file.
  // Note: To save memory with longer traces, you can tell Perfetto to write
  // directly into a file by passing a file descriptor into Setup() above.
  std::ofstream output;
  output.open("example.pftrace", std::ios::out | std::ios::binary);
  output.write(&trace_data[0], std::streamsize(trace_data.size()));
  output.close();
  PERFETTO_LOG("Trace written in example.pftrace file. To read this trace in "
               "text form, run `./tools/traceconv text example.pftrace`");
}

void PerfettoSigleton::StopSystemTracing() {
  // Make sure the last event is closed for this example.
  perfetto::TrackEvent::Flush();
  PERFETTO_LOG("System tracing stopped.");
}