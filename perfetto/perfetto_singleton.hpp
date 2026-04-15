#include <iostream>
#include <fstream>
#include <condition_variable>
#include <memory>

#include "sdk/perfetto.h"

// The set of track event categories that the example is using.
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("rendering")
        .SetDescription("Rendering and graphics events"),
    perfetto::Category("network.debug")
        .SetTags("debug")
        .SetDescription("Verbose network events"),
    perfetto::Category("MNN")
        .SetTags("MNN debug")
        .SetDescription("Detailed MNN events"));

class Observer : public perfetto::TrackEventSessionObserver {
public:
  Observer() {
    perfetto::TrackEvent::AddSessionObserver(this);
  }
  ~Observer() override {
    perfetto::TrackEvent::RemoveSessionObserver(this);
  }

  void OnStart(const perfetto::DataSourceBase::StartArgs &) override {
    std::unique_lock<std::mutex> lock(mutex);
    cv.notify_one();
  }

  void WaitForTracingStart() {
    PERFETTO_LOG("Waiting for tracing to start...");
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [] { return perfetto::TrackEvent::IsEnabled(); });
    PERFETTO_LOG("Tracing started");
  }

private:
  std::mutex mutex;
  std::condition_variable cv;
};

class PerfettoSigleton {
public:
  PerfettoSigleton(const PerfettoSigleton&) = delete;
  PerfettoSigleton& operator=(const PerfettoSigleton&) = delete;
  PerfettoSigleton(PerfettoSigleton&&) = delete;
  PerfettoSigleton& operator=(PerfettoSigleton&&) = delete;  
  ~PerfettoSigleton() {
    StopSystemTracing();
    // StopInprocessTracing(std::move(tracing_session_));
  }
  static PerfettoSigleton& GetInstance() {
    return instance;
  }
  void WaitForTracingStart() {
    observer.reset(new Observer());
    observer->WaitForTracingStart();
  }

private:
  PerfettoSigleton() {
    InitializePerfetto();
    // tracing_session_ = StartInprocessTracing();
  }
  void InitializePerfetto();
  std::unique_ptr<perfetto::TracingSession> StartInprocessTracing();
  void StopInprocessTracing(std::unique_ptr<perfetto::TracingSession> tracing_session);
  void StopSystemTracing();

private:
  // std::unique_ptr<perfetto::TracingSession> tracing_session_;
  std::unique_ptr<Observer> observer{nullptr};

  static PerfettoSigleton instance;
};