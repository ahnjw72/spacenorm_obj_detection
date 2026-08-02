#ifndef INCLUDE_RESTCLIENT_H_
#define INCLUDE_RESTCLIENT_H_

#include <curl/curl.h>
#include <string>
#include <map>
#include <cstdlib>
#include <algorithm>


using namespace std;

typedef map<string, string> headermap;

typedef struct {
    int code;
    string body;
    headermap headers;
} response;

typedef struct {
    const char* data;
    size_t length;
} upload_object;

class CRestClnt
{
  public:
      CRestClnt();
      ~CRestClnt();
    void clearAuth();
    void setAuth(const string& user,const string& password);
    void setJSONMode(bool bMode);
    response get(const string& url);
    response post(const string& url, const string& ctype,
                         const string& data);
    response put(const string& url, const string& ctype,
                        const string& data);
    response del(const string& url);
    
    response uploadfile(const string& url, const string& filename);
    
    response m_res;
    upload_object m_uplObj;

  private:
    static size_t write_callback(void *ptr, size_t size, size_t nmemb,
                                 void *userdata);
    static size_t header_callback(void *ptr, size_t size, size_t nmemb,
				  void *userdata);
    static size_t read_callback(void *ptr, size_t size, size_t nmemb,
                                void *userdata);

    string &ltrim(string &s) {
      s.erase(s.begin(), find_if(s.begin(), s.end(), not1(std::ptr_fun<int, int>(std::isspace))));
      return s;
    }

    string &rtrim(string &s) {
      s.erase(find_if(s.rbegin(), s.rend(), not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
      return s;
    }

    string &trim(string &s) {
      return ltrim(rtrim(s));
    }
    
    
private:
    string m_clientName;
    string m_userPwd;
    bool m_bJSONMode;

};

#endif  // INCLUDE_RESTCLIENT_H_

