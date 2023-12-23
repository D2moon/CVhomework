#include <io.h>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

void readImgs(std::vector<cv::Mat> &imgs, std::string path)
{
    struct _finddata_t fileData;
    intptr_t findHandle = _findfirst((path + "/*.jpg").c_str(), &fileData);

    if (findHandle != -1) {
        do {
            cv::Mat img = cv::imread(path + "/" + fileData.name);
            if (img.empty()) {
                std::cerr << "Failed to read image: " << fileData.name << std::endl;
                continue;
            }
            imgs.push_back(img);
        } while (_findnext(findHandle, &fileData) == 0);
        _findclose(findHandle);
    } else {
        std::cerr << "No matching files found in the directory" << std::endl;
    }
}

cv::Mat RANSAC(std::vector<cv::KeyPoint> keyPoint1, std::vector<cv::KeyPoint> keyPoint2, std::vector<cv::DMatch> matches,
    cv::Mat img1, cv::Mat img2, int turns, double standard=5.0)
{
    cv::Mat BestTr;
    int best = 0;
    for(int turn = 0; turn < turns; turn++) {
        srand(static_cast<unsigned>(time(nullptr)));
        std::vector<int> unique_set;
        while(unique_set.size() < 4) {
            int tmp = rand()%matches.size();
            if(std::find(unique_set.begin(), unique_set.end(), tmp) != unique_set.end()) {
                unique_set.push_back(tmp);
            }
        }
        // Tr*St = En
        cv::Mat St = cv::Mat::ones(3, 4, CV_32F);
        cv::Mat Tr;
        cv::Mat En = cv::Mat::ones(3, 4, CV_32F);
        for(int i = 0; i < 4; i++) {
            cv::DMatch match = matches[i];
            cv::KeyPoint p1 = keyPoint1[match.queryIdx];
            cv::KeyPoint p2 = keyPoint2[match.trainIdx];
            St.at<float>(0, i) = (float)(2*p1.pt.x-img1.cols) / img1.cols;
            St.at<float>(1, i) = (float)(2*p1.pt.y-img1.rows) / img1.rows;
            En.at<float>(0, i) = (float)(2*p2.pt.x-img2.cols) / img2.cols;
            En.at<float>(1, i) = (float)(2*p2.pt.y-img2.rows) / img2.rows;
        }
        cv::Mat StInv;
        cv::invert(St, StInv, cv::DECOMP_SVD);
        Tr = En * StInv;

        int count = 0;
        for(int i = 0; i < matches.size(); i++) {
            cv::DMatch match = matches[i];
            cv::KeyPoint p1 = keyPoint1[match.queryIdx];
            cv::KeyPoint p2 = keyPoint2[match.trainIdx];
            float x1 = Tr.at<float>(0, 0)*p1.pt.x + Tr.at<float>(0, 1)*p1.pt.y + Tr.at<float>(0, 2);
            float y1 = Tr.at<float>(1, 0)*p1.pt.x + Tr.at<float>(1, 1)*p1.pt.y + Tr.at<float>(1, 2);
            cv::Mat vector1 = (cv::Mat_<float>(3, 1) << x1, y1, 1.0);
            cv::Mat vector2 = (cv::Mat_<float>(3, 1) << p2.pt.x, p2.pt.y, 1.0);
            double dis = cv::norm(vector1, vector2);
            if(dis < standard) count++;
            std::cout << dis << std::endl;
        }
        if(count > best) {
            best = count;
            BestTr = Tr;
        }
    }
    return BestTr;
}

void matchImgs(cv::Mat img1, cv::Mat img2, int maxFeatures = 500)
{
    cv::Ptr<cv::SIFT> sift1 =  cv::SIFT::create();
    cv::Ptr<cv::SIFT> sift2 =  cv::SIFT::create();
    std::vector<cv::KeyPoint> keyPoint1, keyPoint2;
    cv::Mat descriptor1, descriptor2;
    std::vector<cv::DMatch> matches;
    sift1->setNFeatures(maxFeatures);
    sift1->detectAndCompute(img1, cv::Mat(), keyPoint1, descriptor1);
    sift2->setNFeatures(maxFeatures);
    sift2->detectAndCompute(img2, cv::Mat(), keyPoint2, descriptor2);
    for(int i = 0; i < descriptor1.rows; i++) {
        int loc = 0;
        double minDis = 10000000.0, secDis = 10000000.0, rate = 0.7;
        cv::Mat vector1 = descriptor1.row(i).clone();
        for(int j = 0; j < descriptor2.rows; j++) {
            cv::Mat vector2 = descriptor2.row(j).clone();
            double dis = cv::norm(vector1, vector2);
            if(dis < minDis) {
                loc = j;
                secDis = minDis;
                minDis = dis;
            }
            else if(dis < secDis) {
                secDis = dis;
            }
        }
        if(minDis < rate * secDis) {
            matches.push_back(cv::DMatch(i, loc, minDis));
        }
    }
    cv::Mat matchResult;
    cv::drawMatches(img1, keyPoint1, img2, keyPoint2, matches, matchResult);
    cv::namedWindow("MyWindow", cv::WINDOW_NORMAL);
    cv::resizeWindow("MyWindow", 6144, 2048);
    cv::imshow("MyWindow", matchResult);
    cv::waitKey();
}

int main()
{
    std::vector<std::string> files = {"/data1"};
    for(int fileId = 0; fileId < files.size(); fileId++) {
        std::vector<cv::Mat> imgs;
        std::string path = "../data" + files[fileId];
        readImgs(imgs, path);
        cv::Mat totalImg = imgs[0];
        for(int imgId = 1; imgId < 2; imgId++) {
            matchImgs(totalImg, imgs[imgId]);
        }
    }


    // cv::Mat img = cv::imread("../data/data1/112_1298.JPG");
    // cv::Ptr<cv::SIFT> sift =  cv::SIFT::create();
    // std::vector<cv::KeyPoint> keypoints;
    // cv::Mat descriptors;

    // sift->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
    // std::cout << keypoints.size() << ' ' << descriptors.rows << std::endl;
    // std::cout << descriptors.row(0) << std::endl;
    // cv::Mat image_with_keypoints;
    // cv::drawKeypoints(img, keypoints, image_with_keypoints);
    // cv::namedWindow("MyWindow", cv::WINDOW_NORMAL);
    // cv::resizeWindow("MyWindow", 640, 480);
    // cv::imshow("MyWindow", image_with_keypoints);
    // cv::waitKey(0);
}