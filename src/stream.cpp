// Create a GStreamer pipeline to stream a video through our plugin

#include <gst/gst.h>
#include <format>
#include "gstfilter.h"
#include "argh.h"
#include <chrono>
#include <atomic>

std::atomic<uint64_t> frame_count = 0;

static GstPadProbeReturn
cb_have_data (GstPad *pad,
              GstPadProbeInfo *info,
              gpointer user_data)
{
  frame_count++;
  return GST_PAD_PROBE_OK;
}


static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "myfilter", GST_RANK_NONE,
      GST_TYPE_MYFILTER);
}

static void my_code_init ()
{
  gst_plugin_register_static (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    "myfilter",
    "Private elements of my application",
    plugin_init,
    "1.0",
    "LGPL",
    "",
    "",
    "");
}


int main(int argc, char* argv[])
{
  argh::parser cmdl(argc, argv);
  if (cmdl[{"-h", "--help"}])
  {
    g_printerr("Usage: %s --mode=[gpu,cpu] <filename> [--output=output.mp4]\n", argv[0]);
    return 0;
  }

  Parameters params;
  auto method = cmdl("mode", "cpu").str();
  auto filename = cmdl(1).str();
  auto output = cmdl({"-o", "--output"}, "").str();
  if (method == "cpu")
    params.device = e_device_t::CPU;
  else if (method == "gpu")
    params.device = e_device_t::GPU;
  else
  {
    g_printerr("Invalid method: %s\n", method.c_str());
    return 1;
  }

  g_debug("Using method: %s", method.c_str());
  gst_init(&argc, &argv);
  my_code_init();
  cpt_init(&params);

  // Create a GStreamer pipeline to stream a video through our plugin
  const char* pipe_str;
  g_print("Output: %s\n", output.c_str());
  if (output.empty())
    pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter name=filter ! videoconvert ! fpsdisplaysink sync=false";
  else
    pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! myfilter name=filter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink name=fdst";



  GError *error = NULL;
  auto pipeline = gst_parse_launch(pipe_str, &error);
  if (!pipeline)
  {
    g_printerr("Failed to create pipeline: %s\n", error->message);
    return 1;
  }

  auto filesrc = gst_bin_get_by_name (GST_BIN (pipeline), "fsrc");
  g_object_set (filesrc, "location", filename.c_str(), NULL);
  g_object_unref (filesrc);

  if (!output.empty())
  {
    auto filesink = gst_bin_get_by_name (GST_BIN (pipeline), "fdst");
    g_object_set (filesink, "location", output.c_str(), NULL);
    g_object_unref (filesink);
  }

  auto filter = gst_bin_get_by_name(GST_BIN(pipeline), "filter");
  if (filter) {
    auto pad = gst_element_get_static_pad(filter, "src");
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, cb_have_data, NULL, NULL);
    gst_object_unref(pad);
    gst_object_unref(filter);
  }

  auto start = std::chrono::steady_clock::now();

  // Start the pipeline
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  // Wait until error or EOS
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE, (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  g_print("Total frames: %lu\n", frame_count.load());
  g_print("Time elapsed: %.2f s\n", elapsed.count());
  g_print("Average FPS: %.2f\n", frame_count.load() / elapsed.count());

  // Free resources
  if (msg != nullptr)
    gst_message_unref(msg);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);

  return 0;
}
