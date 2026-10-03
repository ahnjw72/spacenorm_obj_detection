The API "dev.contextmatter.com/api/v1/vision_nodes/sync_data" accessed by the following command returns sensors and cctvs configuration information for a site mapped by 'access token' and'vision node id'. It is formatted in JSON.

$ curl -X GET --header 'Accept: application/json' --header 'Authorization: Bearer 8OMaemAPcGxvEx_5sa7khnl3FJQnJBUWM3RXBf6z3io' 'https://dev.contextmatter.com/api/v1/vision_nodes/sync_data?ids=5'

I will call the new sensors & CCTV configuration method using this API as "dynamic". On the contrary, currently existing configuration method using json files and *.keys under ./spacenorm_cfg/cctv will be called as "static".

With that in mind, I want to add the "dynamic" sensors & CCTV configuration procedure in current spacenorm_obj_detection project with the following requirements:

1. The json configuration files under ./spacenorm_cfg/behavior and ./spacenorm_cfg/behavior/overrides are still used in both "static" and "dynamic" methods. The ./spacenorm_cfg/behavior/default.json has new keys named "config_method", "config_renew_period", "gateway_id", "access_token" and "vision_node_id" under "cctv_configuration" section like this:

  "cctv_configuration":  {
    "config_method": "static",
    "config_renew_period": -1,
    "gateway_id": null,
    "access_token": null,
    "vision_node_id": null
  }
, where their actual values should be defined in each company or site's specific json file under ./spacenorm_cfg/behavior/overrides.

2. If "config_method" is defined as "dynamic", all the information in the json files and *.keys under .spacenorm_cfg/cctv/ are ignored. The "access_token" and "vision_node_id" are used to specify a specific site when consulting with the API. If "config_method" is "static", the other keys under "cctv_configuration" are ignored and the configuration procedure for the sensors and cctvs remains the same as the current implementation.

3. In the json resonse of the API, which you have to investigate by running for yourslef the above curl command, the members under "sensors" defines virtual human occupancy detectors like this:

  {
      "sensor_id": 805,
      "device_id": "e43022eda616",
      "cctv_id": 440,
      "vision_node_id": 2,
      "mqtt_broker_addr": "localhost",
      "mqtt_broker_port": "1883"
  }

Among them, the "cctv_id" points to the "id" of the members under "cctvs" for specifying a CCTV it watches. The keys "mqtt_broker_addr" and "mqtt_broker_port" are for connecting to an MQTT broker for reporting the detection result. If they are null or empty string(""), MQTT is not used for reporting. The "device_id" is the same as in "static" method. The value of the "vision_node_id" can be used for checking if the sensor belong to the site currently consulting for. The "sensor_id" is not used for now.

4. The members under "cctvs" section has the information about connecting specific CCTVs for human occupancy detection like this:

  {
      "id": 440,
      "url": "rtsp://cheilacc-ansung.spacenorm.com:558/LiveChannel/6/media.smp/profile=2",
      "desc": "EVOL 10x25 Robot #3",
      "domain": "cheilacc-ansung.spacenorm.com",
      "port": "80",
      "channel": "6",
      "snapshot_url": null,
      "snapshot_update_required": true,
      "private_url": "rtsp://192.168.200.13:554/0/profile2/media.smp",
      "min_obj_size_ratio": "1.0",
      "roi_vertices": "[[[0.0,0.14],[0.0,0.8],[0.15,1.0],[1.0,1.0],[1.0,0.34],[0.94,0.34],[0.93,0.23],[0.81,0.23],[0.78,0.0],[0.32,0.0],[0.16,0.13],[0.11,0.3]]]"
  }

Among them, "private_url" should be used to connect to the CCTV through RTSP. The "roi_vertices" is composed of pair of 'ratio's to the width and height of the image to define the ROIs. That scheme is different from the current definition of ROI vertices where img_w, img_h and absolute coordinates are used. The roi_vertices are assumed to be sorted counter-clockwise (as in case "vertices_sorted":1 in current json configuration for each CCTV). I.e. in "dynamic" method, there is no "vertices_sorted", which means that ROI checking assumes the vertices are sorted counter-clockwise by default. If the key "min_obj_size_ratio" is defined, a detected box whose area ratio in percent is smaller than it is ignored. If the key "snapshot_update_required" is set to be true, a snapshot for the CCTV should be captured and uploaded to "snapshot_url" during configuration. If the "snapshot_url" is null, ignore snapshot uploading with a relevant warning logs. In the case of "snapshot_update_required" being true, the future renewal of the configuration will provide the actual information on the roi_vertices.

5. When the "config_method" is set as "dynamic", there should be a background thread to periodically refresh or renew the configuration info using the API and if there is any discrepancy with existing configuration it should be applied to the current running detector threads. I.e., newly defined sensor or CCTV launches new thread and removed sensor or CCTV removes corresponding thread. New roi_veritices info also should be applied to currently running detector threads. Period for the renewal (in the unit of minutes) is set in ./spacenorm_cfg/behavior/overrides/<site>.json with a key named "config_renew_period" under "cctv_configuration". Of course, "config_renew_period" is meaningless in "static" method.

6. Currently, the API's response does not contain the ID and PASSWORD information required for accessing some CCTVs through RTSP. They will be acquired by another API in the future and temporarily the hardcoded ID of 'space'and PASSWORD of 'spacenorm12!@#' for all the CCTVs RTSP connection are used. (If there will be any problem in providing ID and PASSWORD to a CCTV that does not require them, please tell me.)

7. When a detector's configuration is intialized or renewed, a sample image file for each sensor with the ROI drawn should be written to an appropriate folder so that correct ROI borderline will be checked afterwards by human.

8. Current project use *.keys files to test only part of all the occupancy sensors. I don't want the separate *.keys to be used any more in "dynamic" method, but want to devise similar mechanism in case of "dynamic" method too.

9. In "dynamic" method, sensor key string like "정양산업_2F_201" defined in *.keys in "static" method is constructed by "company_name" and "desc" under "cctvs".

10. For the time being, ignore the "security_regions" in the json response of the API.

11. I want to implement this whole feature in a new git branch. After verifying the implementation, it will be merged to the master branch.