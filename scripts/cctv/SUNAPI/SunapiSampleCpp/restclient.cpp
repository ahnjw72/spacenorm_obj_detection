#include "restclient.h"
#include <cstring>
#include <string>
#include <iostream>
#include <map>
#include <sys/stat.h>


CRestClnt::CRestClnt(){
    m_clientName = "HanwhaClient";
    m_bJSONMode = false;
}

CRestClnt::~CRestClnt(){
    
}

void CRestClnt::clearAuth(){
  m_userPwd.clear();
}

void CRestClnt::setAuth(const string& user,const string& password){
  m_userPwd.clear();
  m_userPwd += user+":"+password;
}


void CRestClnt::setJSONMode(bool bMode){
  m_bJSONMode = bMode;     
}
 
response CRestClnt::get(const string& url)
{
  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if (curl)
  {
    if(m_userPwd.length()>0){
      curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY );
      curl_easy_setopt(curl, CURLOPT_USERPWD, m_userPwd.c_str());
    }
    
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1500);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, m_clientName.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CRestClnt::write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, CRestClnt::header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA,(void*) this);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
    if (m_bJSONMode == true) 
    {
            string acceptApp = "Accept: application/json";
            curl_slist* header = NULL;
            header = curl_slist_append(header, acceptApp.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header);
     }
    
    res = curl_easy_perform(curl);
    if (res != 0)
    {
      m_res.body = "Failed to query.";
      m_res.code = -1;
      return m_res;
    }
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    m_res.code = static_cast<int>(http_code);

    curl_easy_cleanup(curl);
  }

  return m_res;
}

response CRestClnt::post(const string& url,
                                      const string& ctype,
                                      const string& data)
{
 
  response ret;
  string ctype_header = "Content-Type: " + ctype;
  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if (curl)
  {
    if(m_userPwd.length()>0){
      curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY );
      curl_easy_setopt(curl, CURLOPT_USERPWD, m_userPwd.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 1500);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, m_clientName.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, data.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CRestClnt::write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, CRestClnt::header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
    curl_slist* header = NULL;
    header = curl_slist_append(header, ctype_header.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header);
    res = curl_easy_perform(curl);
    if (res != 0)
    {
      ret.body = "Failed to query.";
      ret.code = -1;
      return ret;
    }
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    ret.code = static_cast<int>(http_code);

    curl_easy_cleanup(curl);
  }

  return ret;
}

response CRestClnt::put(const string& url,
                                     const string& ctype,
                                     const string& data)
{
 
  response ret;
  string ctype_header = "Content-Type: " + ctype;
  upload_object up_obj;
  up_obj.data = data.c_str();
  up_obj.length = data.size();

  CURL *curl;
  CURLcode res;

  curl = curl_easy_init();
  if (curl)
  {
    if(m_userPwd.length()>0){
      curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY );
      curl_easy_setopt(curl, CURLOPT_USERPWD, m_userPwd.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_USERAGENT, m_clientName.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_PUT, 1L);
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, CRestClnt::read_callback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &up_obj);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CRestClnt::write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, CRestClnt::header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
    curl_easy_setopt(curl, CURLOPT_INFILESIZE,
                     static_cast<long>(up_obj.length));

   
    curl_slist* header = NULL;
    header = curl_slist_append(header, ctype_header.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, header);
    
    res = curl_easy_perform(curl);
    if (res != 0)
    {
      ret.body = "Failed to query.";
      ret.code = -1;
      return ret;
    }
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    ret.code = static_cast<int>(http_code);

    curl_easy_cleanup(curl);
  }

  return ret;
}

response CRestClnt::del(const string& url)
{
  response ret;
  const char* http_delete = "DELETE";
  CURL *curl;
  CURLcode res;
  
  curl = curl_easy_init();
  if (curl)
  {
    if(m_userPwd.length()>0){
      curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY );
      curl_easy_setopt(curl, CURLOPT_USERPWD,m_userPwd.c_str());
    }
   
    curl_easy_setopt(curl, CURLOPT_USERAGENT,m_clientName.c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, http_delete);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CRestClnt::write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, CRestClnt::header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, (void*) this);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
   
    res = curl_easy_perform(curl);
    if (res != 0)
    {
      ret.body = "Failed to query.";
      ret.code = -1;
      return ret;
    }
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    ret.code = static_cast<int>(http_code);

    curl_easy_cleanup(curl);
  }

  return ret;
}


response CRestClnt::uploadfile(const string& url, const string& filename){
    CURL *curl;
  CURLcode res;
  response ret;
  
  double upload_speed, tot_time;
  FILE *fd;
  struct stat file_info;
  fd = fopen(filename.c_str(), "rb"); 
  if(!fd) {
      ret.code = -1;
      ret.body = "Error in opening file";
    return ret; 
  }
 
  
  if(fstat(fileno(fd), &file_info) != 0) {
    ret.code = -1;
      ret.body = "Error in getting file size";
    return ret; 
  }
 
  curl = curl_easy_init();
  if(curl) {
    if(m_userPwd.length()>0){
      curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_ANY );
      curl_easy_setopt(curl, CURLOPT_USERPWD,m_userPwd.c_str());
    }
    curl_easy_setopt(curl, CURLOPT_USERAGENT,m_clientName.c_str());
    curl_easy_setopt(curl, CURLOPT_URL,url.c_str());
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_READDATA, fd);
    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE,(curl_off_t)file_info.st_size);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
    res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
      ret.code = -1;
      ret.body = curl_easy_strerror(res);
 
    }
    else {
    
      curl_easy_getinfo(curl, CURLINFO_SPEED_UPLOAD, &upload_speed);
      curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME, &tot_time);
 
      fprintf(stderr, "Upload Speed: %.3f bytes/sec Total time taken %.3f seconds\n",
              upload_speed, tot_time);
         ret.code = 0;
         ret.body = "Success in sending files";
  
    }
  
    curl_easy_cleanup(curl);
  }
  return ret;
}

size_t CRestClnt::write_callback(void *data, size_t size, size_t nmemb,
                            void *userdata)
{
  CRestClnt* r;
  r = reinterpret_cast<CRestClnt*>(userdata);
  if(r == NULL)
      return 0;
  r->m_res.body.append(reinterpret_cast<char*>(data), size*nmemb);
  //fprintf(stderr,">>>>%s\n",(char*) data);
  return (size * nmemb);
}


size_t CRestClnt::header_callback(void *data, size_t size, size_t nmemb,
                            void *userdata)
{
  CRestClnt* r;
  r = reinterpret_cast<CRestClnt*>(userdata);
  if(r == NULL)
      return 0;
  std::string header(reinterpret_cast<char*>(data), size*nmemb);
  size_t seperator = header.find_first_of(":");
  if ( std::string::npos == seperator ) { 
    //roll with non seperated headers... 
    r->trim(header); 
    if ( 0 == header.length() ){ 
	return (size * nmemb); //blank line;
    } 
    r->m_res.headers[header] = "present";
  } else {
    std::string key = header.substr(0, seperator);
    r->trim(key);
    std::string value = header.substr(seperator + 1);
    r->trim (value);
     r->m_res.headers[key] = value;
  }

  return (size * nmemb);
}


size_t CRestClnt::read_callback(void *data, size_t size, size_t nmemb,
                            void *userdata)
{
  CRestClnt* r;
  r = reinterpret_cast<CRestClnt*>(userdata);
  if(r == NULL)
      return 0;
 
  size_t curl_size = size * nmemb;
  size_t copy_size = (r->m_uplObj.length < curl_size) ? r->m_uplObj.length : curl_size;
 
  memcpy(data, r->m_uplObj.data, copy_size);
  
  r->m_uplObj.length -= copy_size;
  r->m_uplObj.data += copy_size;
 
  return copy_size;
}
