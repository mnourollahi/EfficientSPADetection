#ifndef SOCIAL_NETWORK_MICROSERVICES_SRC_USERTIMELINESERVICE_USERTIMELINEHANDLER_H_
#define SOCIAL_NETWORK_MICROSERVICES_SRC_USERTIMELINESERVICE_USERTIMELINEHANDLER_H_

#include <iostream>
#include <string>

#include <mongoc.h>
#include <bson/bson.h>

#include "../../gen-cpp/UserTimelineService.h"
#include "../../gen-cpp/PostStorageService.h"
#include "../logger.h"
#include "../tracing.h"
#include "../ClientPool.h"
#include "../RedisClient.h"
#include "../ThriftClient.h"

namespace social_network {

class UserTimelineHandler : public UserTimelineServiceIf {
 public:
  UserTimelineHandler(
      ClientPool<RedisClient> *,
      mongoc_client_pool_t *,
      ClientPool<ThriftClient<PostStorageServiceClient>> *);
  ~UserTimelineHandler() override = default;


  void WriteUserTimeline(int64_t, int64_t, int64_t, int64_t,
                         const std::map<std::string, std::string> &) override;																		   
  void ReadUserTimeline(std::vector<Post> &, int64_t, int64_t, int, int,
                        const std::map<std::string, std::string> &) override ;												

 private:
  ClientPool<RedisClient> *_redis_client_pool;
  mongoc_client_pool_t *_mongodb_client_pool;
  ClientPool<ThriftClient<PostStorageServiceClient>> *_post_client_pool;
};

UserTimelineHandler::UserTimelineHandler(
    ClientPool<RedisClient> *redis_pool,
    mongoc_client_pool_t *mongodb_pool,
    ClientPool<ThriftClient<PostStorageServiceClient>> *post_client_pool) {
  _redis_client_pool = redis_pool;
  _mongodb_client_pool = mongodb_pool;
  _post_client_pool = post_client_pool;
}

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
    int start,
    int stop,
    const std::map<std::string, std::string> &carrier) {

  // Obtain a PostStorageServiceClient instance from `_post_client_pool`
  auto post_client_wrapper = _post_client_pool->Pop();
  if (!post_client_wrapper) {
    ServiceException se;
    se.errorCode = ErrorCode::SE_THRIFT_CONN_ERROR;
    se.message = "Failed to connect to post-storage-service";
    throw se;
  }
  auto post_client = post_client_wrapper->GetClient();

  try {
    post_client->ReadUserTimelinePostStorage(_return, req_id, user_id, start, stop, carrier);
  } catch (...) {
    _post_client_pool->Push(post_client_wrapper);
    LOG(error) << "Failed to delegate ReadUserTimelinePostStorage to PostStorageHandler";
    throw;
  }
  _post_client_pool->Push(post_client_wrapper);
}

}  // namespace social_network

#endif  // SOCIAL_NETWORK_MICROSERVICES_SRC_USERTIMELINESERVICE_USERTIMELINEHANDLER_H_

																					
