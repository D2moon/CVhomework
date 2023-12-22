#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int height, width;

std::vector<std::string> readFileNames(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<std::string> values;
    std::string str;
    while (std::getline(file, str)) {
        values.push_back(str);
    }

    file.close();
    return values;
}

std::vector<cv::Vec3f> readFloats(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<cv::Vec3f> values;
    cv::Vec3f v;
    while (file >> v[0] >> v[1] >> v[2]) {
        values.push_back(v);
    }

    file.close();
    return values;
}

cv::Mat PSshadows(cv::Mat Mask, cv::Mat Lights, std::vector<cv::Mat> Imgs, int NUM_IMGS, int use, cv::Mat &Albedo)
{
    cv::Mat LightsInv;
    cv::Mat Normals(height, width, CV_32FC3, cv::Scalar::all(0));

    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) if(Mask.at<uchar>(cv::Point(i, j)) > 0){
            std::vector<std::pair<float, int> > Iv;
            for(int k = 0; k < NUM_IMGS; k++) {
                Iv.push_back(std::make_pair(Imgs[k].at<uchar>(cv::Point(i, j))/255.0f, k));
            }
            std::sort(Iv.begin(), Iv.end(), [](const auto& a, const auto& b){
                return a.first < b.first;
            });

            cv::Mat selectLights(use, 3, CV_32F);
            std::vector<float> selectI;
            int st = (NUM_IMGS-use)/2;
            for(int k = 0; k < use; k++) {
                int now = st+k;
                Lights.row(Iv[now].second).copyTo(selectLights.row(k));
                selectI.push_back(Iv[now].first);
            }

            selectLights = selectLights.t();
            cv::invert(selectLights, LightsInv, cv::DECOMP_SVD);
            cv::Mat I = cv::Mat(selectI).reshape(1, 1);
            cv::Mat n = I * LightsInv;
            float p = sqrt(n.dot(n));
            if(p > 0) n = n/p;
            if(n.at<float>(2,0) == 0) {n.at<float>(2,0) = 1.0;}
            Normals.at<cv::Vec3f>(cv::Point(i, j)) = n;
            Albedo.at<float>(cv::Point(i, j)) = p;
        }
    }
    return Normals;
}

cv::Mat reRender(cv::Mat Mask, cv::Mat Albedo, cv::Mat Light, cv::Mat Normals)
{
    cv::Mat Result(height, width, CV_32FC1, cv::Scalar::all(0.0));
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) if(Mask.at<uchar>(cv::Point(i, j)) > 0) {
            cv::Mat LightRow = Light.clone().t();
            float dot = Normals.at<cv::Vec3f>(cv::Point(i, j)).dot(LightRow);
            float albedo = Albedo.at<float>(cv::Point(i, j));
            Result.at<float>(cv::Point(i, j)) = dot * albedo;
        }
    }
    return Result;
}

void dealMask(cv::Mat Mask, cv::Mat &Normalmap, cv::Mat &NormalRender)
{
    for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; j++) {
            if(Mask.at<uchar>(cv::Point(i, j)) <= 0){
                cv::Vec3b nullvec(0, 0, 0);
                Normalmap.at<cv::Vec3b>(cv::Point(i, j)) = nullvec;
                NormalRender.at<uchar>(cv::Point(i, j)) = 0;
            }
        }
    }
}

int main() 
{
    std::string dataFormat = "PNG";
    std::vector<std::string> dataNameStack = {"/bear", "/cat", "/pot", "/buddha"};

    for (int testId = 0; testId < 4; ++testId) {
        std::string dataName = dataNameStack[testId] + dataFormat;
        std::string dataDir = "../pmsData" + dataName;
        std::vector<std::string> fileNames = readFileNames(dataDir+"/filenames.txt");
        int NUM_IMGS = fileNames.size();

        cv::Mat Lights(NUM_IMGS, 3, CV_32F);
        cv::Mat Mask = cv::imread(dataDir+"/mask.png", cv::IMREAD_GRAYSCALE);
        height = Mask.rows;
        width = Mask.cols;
        std::vector<cv::Mat> Imgs;
        std::vector<cv::Vec3f> light = readFloats(dataDir+"/light_directions.txt");
        std::vector<cv::Vec3f> intensity = readFloats(dataDir+"/light_intensities.txt");

        for(int i = 0; i < NUM_IMGS; i++) {
            Lights.at<float>(i, 0) = light[i][0];
            Lights.at<float>(i, 1) = light[i][1];
            Lights.at<float>(i, 2) = light[i][2];
            cv::Mat img = cv::imread(dataDir+'/'+fileNames[i], cv::IMREAD_COLOR);

            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            channels[0] /= intensity[i][0];
            channels[1] /= intensity[i][1];
            channels[2] /= intensity[i][2];
            cv::merge(channels, img);
            cv::Mat grayImg;
            cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
            Imgs.push_back(grayImg);
        }

        cv::Mat Albedo(height, width, CV_32F, cv::Scalar::all(0.0));
        cv::Mat Normals = PSshadows(Mask, Lights, Imgs, NUM_IMGS, 50, Albedo);
        cv::Mat Normalmap, Normalalbedo, NormalRender;
        cv::normalize(Albedo, Normalalbedo, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::cvtColor(Normals, Normalmap, cv::COLOR_BGR2RGB);
        cv::normalize(Normalmap, Normalmap, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat FrontLight = (cv::Mat_<float>(1, 3) << 0.0, -1.0, 0.0);
        cv::Mat Render = reRender(Mask, Albedo, FrontLight, Normals);
        cv::normalize(Render, NormalRender, 0, 255, cv::NORM_MINMAX, CV_8U);

        dealMask(Mask, Normalmap, NormalRender);
        cv::imwrite("./"+dataNameStack[testId]+"Albedo.png", Normalalbedo);
        cv::imwrite("./"+dataNameStack[testId]+"Norm.png", Normalmap);
        cv::imwrite("./"+dataNameStack[testId]+"Render.png", NormalRender);
    }
    return 0;   
}
