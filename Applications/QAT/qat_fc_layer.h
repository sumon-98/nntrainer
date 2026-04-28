#ifndef __QAT_FC_LAYER_H__
#define __QAT_FC_LAYER_H__

#include <layer_context.h>
// #include <layer_devel.h>
#include <layer_impl.h>  // Changed from layer_devel.h
#include <node_exporter.h>
#include <utility>
#include <array>
#include <map>
#include <mutex>
#include <tensor_dim.h>

namespace nntrainer {

class QATFullyConnectedLayer final : public Layer {
public:
  // Static registry to track QAT layers by name
  static std::map<std::string, QATFullyConnectedLayer*>& getRegistry() {
    static std::map<std::string, QATFullyConnectedLayer*> registry;
    return registry;
  }
  
  static std::mutex& getRegistryMutex() {
    static std::mutex mtx;
    return mtx;
  }
  
  static QATFullyConnectedLayer* getLayerByName(const std::string& name) {
    std::lock_guard<std::mutex> lock(getRegistryMutex());
    auto it = getRegistry().find(name);
    if (it != getRegistry().end()) {
      return it->second;
    }
    return nullptr;
  }

  QATFullyConnectedLayer() : Layer(), running_min_val(0.0f), running_max_val(0.0f), initialized_stats(false), a_scale(0.0f), a_zp(0.0f), w_scale(0.0f), w_zp(0.0f), unit(0), disable_bias(false) {
    weight_idx.fill(std::numeric_limits<unsigned>::max());
  }
  
  
  
  ~QATFullyConnectedLayer() {
    // Remove from registry on destruction
    std::lock_guard<std::mutex> lock(getRegistryMutex());
    for (auto it = getRegistry().begin(); it != getRegistry().end(); ++it) {
      if (it->second == this) {
        getRegistry().erase(it);
        break;
      }
    }
  }

  void finalize(InitLayerContext &context) override;
  void forwarding(RunLayerContext &context, bool training) override;
  void calcDerivative(RunLayerContext &context) override;
  void calcGradient(RunLayerContext &context) override;
  bool supportBackwarding() const override { return true; }
  void exportTo(Exporter &exporter, const ml::train::ExportMethods &method) const override {}
  void setProperty(const std::vector<std::string> &values) override;

  const std::string getType() const override { return type; }

  inline static const std::string type = "qat_fully_connected";

  // Helpers for printing scale/zp out
  float getActScale() const { return a_scale; }
  float getActZeroPoint() const { return a_zp; }
  float getWeightScale() const { return w_scale; }
  float getWeightZeroPoint() const { return w_zp; }

private:
  float running_min_val;
  float running_max_val;
  bool initialized_stats;

  float a_scale, a_zp;
  float w_scale, w_zp;

  unsigned int unit;
  bool disable_bias;
  std::string layer_name;  // Store the name

  std::array<unsigned int, 2> weight_idx;
  enum FCParams { weight, bias };

  const float momentum = 0.1f;
  const float q_min_a = 0.0f;
  const float q_max_a = 255.0f;
  const float q_min_w = -128.0f;
  const float q_max_w = 127.0f;

  Tensor fakeQuantize(const Tensor &x, float &min_val, float &max_val, float q_min, float q_max, bool update_stats, float &out_scale, float &out_zp);
  
  void registerLayer(const std::string& name) {
    std::lock_guard<std::mutex> lock(getRegistryMutex());
    getRegistry()[name] = this;
    layer_name = name;
  }
};

} // namespace nntrainer

#endif
