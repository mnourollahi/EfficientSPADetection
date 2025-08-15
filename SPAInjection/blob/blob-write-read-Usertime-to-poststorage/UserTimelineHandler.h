#ifndef SOCIAL_NETWORK_MICROSERVICES_SRC_USERTIMELINESERVICE_USERTIMELINEHANDLER_H_
#define SOCIAL_NETWORK_MICROSERVICES_SRC_USERTIMELINESERVICE_USERTIMELINEHANDLER_H_


#include <iostream>
#include <string>
#include <map>

#include <mongoc.h>
#include <bson/bson.h>

#include "../../gen-cpp/UserTimelineService.h"
#include "../../gen-cpp/PostStorageService.h"
#include "../logger.h"
#include "../tracing.h"
#include "../ClientPool.h"
#include "../RedisClient.h"
#include "../ThriftClient.h"
#include "PostStorageHandler.h"  // Ensure this is included

namespace social_network {

class UserTimelineHandler : public UserTimelineServiceIf {
 public:
  UserTimelineHandler(
      PostStorageHandler* post_storage_handler,
      ClientPool<RedisClient> *redis_client_pool,
      mongoc_client_pool_t *mongodb_client_pool,
      ClientPool<ThriftClient<PostStorageServiceClient>> *post_client_pool);
  ~UserTimelineHandler() override = default;

  void WriteUserTimeline(int64_t req_id, int64_t post_id, int64_t user_id,
                         int64_t timestamp,
                         const std::map<std::string, std::string> &carrier) override;
  
  void ReadUserTimeline(std::vector<Post> &_return, int64_t req_id, int64_t user_id,
                        const std::map<std::string, std::string> &carrier) override;

 private:
  PostStorageHandler *_post_storage_handler;
  ClientPool<RedisClient> *_redis_client_pool;
  mongoc_client_pool_t *_mongodb_client_pool;
  ClientPool<ThriftClient<PostStorageServiceClient>> *_post_client_pool;
};

// Constructor implementation
UserTimelineHandler::UserTimelineHandler(
    PostStorageHandler* post_storage_handler,
    ClientPool<RedisClient> *redis_client_pool,
    mongoc_client_pool_t *mongodb_client_pool,
    ClientPool<ThriftClient<PostStorageServiceClient>> *post_client_pool)
    : _post_storage_handler(post_storage_handler),
      _redis_client_pool(redis_client_pool),
      _mongodb_client_pool(mongodb_client_pool),
      _post_client_pool(post_client_pool) {}


void UserTimelineHandler::WriteUserTimeline(
    int64_t req_id,
    int64_t post_id,
    int64_t user_id,
    int64_t timestamp,
    const std::map<std::string, std::string> &carrier) {
    
  
  // Initialize a span
  TextMapReader reader(carrier);
  std::map<std::string, std::string> writer_text_map;
  TextMapWriter writer(writer_text_map);
  auto parent_span = opentracing::Tracer::Global()->Extract(reader);
  auto span = opentracing::Tracer::Global()->StartSpan(
      "WriteUserTimelineDeligated",
      { opentracing::ChildOf(parent_span->get()) });
  opentracing::Tracer::Global()->Inject(span->context(), writer);

  // Delegate WriteUserTimeline to PostStorageService's StorePost method
  auto post_client_wrapper = _post_client_pool->Pop();
  if (!post_client_wrapper) {
    ServiceException se;
    se.errorCode = ErrorCode::SE_THRIFT_CONN_ERROR;
    se.message = "Failed to connect to post-storage-service";
    throw se;
  }
  auto post_client = post_client_wrapper->GetClient();
  try {
    // Assuming StorePost will handle both storing the post and updating the timeline
    Post post;
    post.post_id = post_id;
    post.timestamp = timestamp;
    post.creator.user_id = user_id;
    post_client->StorePost(req_id, post, carrier);
  } catch (...) {
    _post_client_pool->Push(post_client_wrapper);
    LOG(error) << "Failed to delegate WriteUserTimeline to post-storage-service";
    throw;
  }
  _post_client_pool->Push(post_client_wrapper);
  span->Finish();
}

void UserTimelineHandler::ReadUserTimeline(
    std::vector<Post> &_return,
    int64_t req_id,
    int64_t user_id,
    const std::map<std::string, std::string> &carrier) {
    
    // Delegate the logic to PostStorageHandler's ReadUserTimeline implementation
    _post_storage_handler->ReadUserTimeline(_return, req_id, user_id, carrier);
}

}  // namespace social_network

#endif  // SOCIAL_NETWORK_MICROSERVICES_SRC_USERTIMELINESERVICE_USERTIMELINEHANDLER_H_

																					
