#ifndef SOCIAL_NETWORK_MICROSERVICES_SRC_HOMETIMELINESERVICE_HOMETIMELINEHANDLER_H_
#define SOCIAL_NETWORK_MICROSERVICES_SRC_HOMETIMELINESERVICE_HOMETIMELINEHANDLER_H_

#include <iostream>
#include <string>
#include <future>

#include <cpp_redis/cpp_redis>


#include "../../gen-cpp/HomeTimelineService.h"
#include "../../gen-cpp/PostStorageService.h"
#include "../logger.h"
#include "../tracing.h"
#include "../ClientPool.h"
#include "../RedisClient.h"
#include "../ThriftClient.h"

namespace social_network {

class HomeTimelineHandler : public HomeTimelineServiceIf {
 public:
  HomeTimelineHandler(
      ClientPool<RedisClient> *redis_pool,
      ClientPool<ThriftClient<PostStorageServiceClient>> *post_client_pool);
  ~HomeTimelineHandler() override = default;

  void ReadHomeTimeline(
      std::vector<Post> &return_posts, int64_t req_id, int64_t user_id,
      int start_idx, int stop_idx,
      const std::map<std::string, std::string> &carrier) override;

 private:
  ClientPool<RedisClient> *_redis_client_pool;
  ClientPool<ThriftClient<PostStorageServiceClient>> *_post_client_pool;
};

HomeTimelineHandler::HomeTimelineHandler(
    ClientPool<RedisClient> *redis_pool,
    ClientPool<ThriftClient<PostStorageServiceClient>> *post_client_pool)
    : _redis_client_pool(redis_pool), _post_client_pool(post_client_pool) {}



void HomeTimelineHandler::ReadHomeTimeline(
    std::vector<Post> &return_posts, int64_t req_id, int64_t user_id,
    int start_idx, int stop_idx,
    const std::map<std::string, std::string> &carrier) {
  auto post_client_wrapper = _post_client_pool->Pop();
  if (!post_client_wrapper) {
    ServiceException se;
    se.errorCode = ErrorCode::SE_THRIFT_CONN_ERROR;
    se.message = "Failed to connect to post-storage-service";
    throw se;
  }

  auto post_client = post_client_wrapper->GetClient();
  try {
    post_client->ReadHomeTimelinePosts(return_posts, req_id, user_id, start_idx,
                                       stop_idx, carrier);
  } catch (...) {
    _post_client_pool->Remove(post_client_wrapper);
    LOG(error) << "Failed to fetch home timeline from post-storage-service";
    throw;
  }
  _post_client_pool->Push(post_client_wrapper);
}



} // namespace social_network

#endif //SOCIAL_NETWORK_MICROSERVICES_SRC_HOMETIMELINESERVICE_HOMETIMELINEHANDLER_H_
