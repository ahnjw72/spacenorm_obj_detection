#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "json.hpp"
#include  <memory>

#include "restclient.h"

using json = nlohmann::json;


//Depends on libcurl to send the sunapi request

int main(int argc, char **argv)
{
    int i;

    if (argc < 4)
    {
        fprintf(stderr, "USAGE:\n testClient <IP> <username> <password>\n");
        return 0;
    }

    std::string ipAddr = argv[1];
    std::string username = argv[2];
    std::string password = argv[3];


    //SAMPLE sunapi request
    std::string sampleURL = "http://"+ipAddr+"/stw-cgi/system.cgi?msubmenu=deviceinfo&action=view";


    std::shared_ptr<CRestClnt> testClient(new CRestClnt());
    testClient->setAuth(username, password);
    //If json mode is set to true response will be in JSON format
    testClient->setJSONMode(true);
    response res;

    res = testClient->get(sampleURL);
    fprintf(stderr, "RET CODE:\n %d\n\n>>>>>>>>>\n\n", res.code);

    //If sucess response
    if (res.code == 200)
    {
        fprintf(stderr, "RET MSG:\n %s\n\n>>>>>>>>>\n\n", res.body.c_str());

        //Parsing the json response
        auto result = json::parse(res.body);
        //Print one of the pared field
        fprintf(stderr, "%s\n", result["Model"].get<std::string>().c_str());

        //Dumping parsed json
        fprintf(stderr, "DUMP: \n %s\n", result.dump(4).c_str());
    }

    return 0;
}