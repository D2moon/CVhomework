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

// Tr*St = En  St表示原图像坐标，En表示变换后坐标
cv::Mat findHomography(cv::Mat St, cv::Mat En)
{
    cv::Mat X;
    cv::Mat A(8, 8, CV_32F);
    cv::Mat B(8, 1, CV_32F);
    for(int i = 0; i < 4; i++) {
        A.at<float>(2*i, 0) = St.at<float>(0, i);
        A.at<float>(2*i, 1) = St.at<float>(1, i);
        A.at<float>(2*i, 2) = 1.0;
        A.at<float>(2*i, 3) = 0.0;
        A.at<float>(2*i, 4) = 0.0;
        A.at<float>(2*i, 5) = 0.0;
        A.at<float>(2*i, 6) = -St.at<float>(0, i) * En.at<float>(0, i);
        A.at<float>(2*i, 7) = -St.at<float>(1, i) * En.at<float>(0, i);
        A.at<float>(2*i+1, 0) = 0.0;
        A.at<float>(2*i+1, 1) = 0.0;
        A.at<float>(2*i+1, 2) = 0.0;
        A.at<float>(2*i+1, 3) = St.at<float>(0, i);
        A.at<float>(2*i+1, 4) = St.at<float>(1, i);
        A.at<float>(2*i+1, 5) = 1.0;
        A.at<float>(2*i+1, 6) = -St.at<float>(0, i) * En.at<float>(1, i);
        A.at<float>(2*i+1, 7) = -St.at<float>(1, i) * En.at<float>(1, i);
        B.at<float>(2*i, 0) = En.at<float>(0, i);
        B.at<float>(2*i, 1) = En.at<float>(1, i);
    }
    cv::solve(A, B, X, cv::DECOMP_SVD);
    cv::Mat Tr(3, 3, CV_32F);
    Tr.at<float>(0, 0) = X.at<float>(0, 0);
    Tr.at<float>(0, 1) = X.at<float>(1, 0);
    Tr.at<float>(0, 2) = X.at<float>(2, 0);
    Tr.at<float>(1, 0) = X.at<float>(3, 0);
    Tr.at<float>(1, 1) = X.at<float>(4, 0);
    Tr.at<float>(1, 2) = X.at<float>(5, 0);
    Tr.at<float>(2, 0) = X.at<float>(6, 0);
    Tr.at<float>(2, 1) = X.at<float>(7, 0);
    Tr.at<float>(2, 2) = 1.0;
    return Tr;
}

cv::Mat RANSAC(std::vector<cv::KeyPoint> keyPoint1, std::vector<cv::KeyPoint> keyPoint2, std::vector<cv::DMatch> matches,
    cv::Mat img1, cv::Mat img2, int turns=1000, double standard=5)
{
    cv::Mat BestTr;
    int best = 0;
    for(int turn = 0; turn < turns; turn++) {
        std::vector<int> unique_set;
        while(unique_set.size() < 4) {
            int tmp = rand()%matches.size();
            // std::cout << tmp << std::endl;
            if(std::find(unique_set.begin(), unique_set.end(), tmp) == unique_set.end()) {
                unique_set.push_back(tmp);
            }
        }
        cv::Mat St = cv::Mat::ones(3, 4, CV_32F);
        cv::Mat En = cv::Mat::ones(3, 4, CV_32F);
        for(int i = 0; i < 4; i++) {
            cv::DMatch match = matches[unique_set[i]];
            cv::KeyPoint p1 = keyPoint1[match.queryIdx];
            cv::KeyPoint p2 = keyPoint2[match.trainIdx];
            St.at<float>(0, i) = p1.pt.x;
            St.at<float>(1, i) = p1.pt.y;
            En.at<float>(0, i) = p2.pt.x;
            En.at<float>(1, i) = p2.pt.y;
            // St.at<float>(0, i) = (float)(2*p1.pt.x-img1.cols) / img1.cols;
            // St.at<float>(1, i) = (float)(2*p1.pt.y-img1.rows) / img1.rows;
            // En.at<float>(0, i) = (float)(2*p2.pt.x-img2.cols) / img2.cols;
            // En.at<float>(1, i) = (float)(2*p2.pt.y-img2.rows) / img2.rows;
        }
        cv::Mat Tr = findHomography(St, En);
        // std::cout << Tr << std::endl;
        int count = 0;
        for(int i = 0; i < matches.size(); i++) {
            cv::DMatch match = matches[i];
            cv::KeyPoint p1 = keyPoint1[match.queryIdx];
            cv::KeyPoint p2 = keyPoint2[match.trainIdx];
            // std::cout << p1.pt.x << ' ' << p1.pt.y << ' ' << p2.pt.x << ' ' << p2.pt.y << std::endl;
            float x1 = Tr.at<float>(0, 0)*p1.pt.x + Tr.at<float>(0, 1)*p1.pt.y + Tr.at<float>(0, 2);
            float y1 = Tr.at<float>(1, 0)*p1.pt.x + Tr.at<float>(1, 1)*p1.pt.y + Tr.at<float>(1, 2);
            float z1 = Tr.at<float>(2, 0)*p1.pt.x + Tr.at<float>(2, 1)*p1.pt.y + Tr.at<float>(2, 2);
            // std::cout << x1 << ' ' << y1 << ' ' << z1 << std::endl;
            // std::cout << x1/z1 << ' ' << y1/z1 << ' ' << z1 << std::endl;
            // std::cout << std::endl;
            cv::Mat vector1 = (cv::Mat_<float>(3, 1) << x1/z1, y1/z1, 1.0);
            cv::Mat vector2 = (cv::Mat_<float>(3, 1) << p2.pt.x, p2.pt.y, 1.0);
            // std::cout << vector1 << '\n' << vector2 << std::endl;
            double dis = cv::norm(vector1, vector2);
            if(dis < standard) count++;
            // std::cout << dis << ' ' << count << std::endl;
        }
        if(count > best) {
            best = count;
            BestTr = Tr;
        }
    }
    std::cout << best << std::endl;
    return BestTr;
}

cv::Mat spliceImgs(cv::Mat img1, cv::Mat img2, cv::Mat Tr)
{
    int L = 0, R = 0, U = 0, D = 0, exWidth = 0, exHeight = 0;
    int originalWidth = img1.cols;
    int originalHeight = img1.rows;
    float img2Width = img2.cols;
    float img2Height = img2.rows;
    cv::Mat TrInv;
    if(!cv::invert(Tr, TrInv, cv::DECOMP_SVD)) {
        std::cerr << "矩阵不可逆!" << std::endl;   
    }
    std::cout << "test6" << std::endl;

    std::vector<cv::Mat> img2Loc;
    cv::Mat LU = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, 1.0f);
    cv::Mat RU = (cv::Mat_<float>(3, 1) << img2Width, 0.0f, 1.0f);
    cv::Mat LD = (cv::Mat_<float>(3, 1) << 0.0f, img2Height, 1.0f);
    cv::Mat RD = (cv::Mat_<float>(3, 1) << img2Width, img2Height, 1.0f);
    img2Loc.push_back(LU);
    img2Loc.push_back(RU);
    img2Loc.push_back(LD);
    img2Loc.push_back(RD);
    for(int i = 0; i < img2Loc.size(); i++) {
        cv::Mat New = TrInv * img2Loc[i];
        float x = New.at<float>(0, 0) / New.at<float>(2, 0);
        float y = New.at<float>(1, 0) / New.at<float>(2, 0);
        if(x < 0) L = int(fabs(x));
        if(x > originalWidth) R = int(x-originalWidth);
        if(y < 0) U = int(fabs(y));
        if(y > originalHeight) D = int(y-originalHeight);
    }
    exWidth = originalWidth + L + R;
    exHeight = originalHeight + U + D;
    cv::Mat exImg(exHeight, exWidth, img1.type());
    img1.copyTo(exImg(cv::Rect(L, U, originalWidth, originalHeight)));

    for(int h = 0; h < exHeight; h++) {
        for(int w = 0; w < exWidth; w++) {
            float blue = exImg.at<cv::Vec3b>(h, w)[0], green = exImg.at<cv::Vec3b>(h, w)[1], red = exImg.at<cv::Vec3b>(h, w)[2];
            if(blue == 0 && green == 0 && red == 0) {
                cv::Mat Loc1 = (cv::Mat_<float>(3, 1) << w-L, h-U, 1.0f);
                cv::Mat Loc2 = Tr * Loc1;
                float w2 = Loc2.at<float>(0, 0) / Loc2.at<float>(2, 0);
                float h2 = Loc2.at<float>(1, 0) / Loc2.at<float>(2, 0);
                if(w2 < 0 || w2 > img2Width-1 || h2 < 0 || h2 > img2Height-1) {
                    continue;
                }
                else {
                    int x1 = w2, y1 = h2;
                    int x2 = x1 + ((w2-x1)>0), y2 = y1 + ((h2-y1)>0);
                    cv::Vec3b v1 = img2.at<cv::Vec3b>(y1, x1), v2 = img2.at<cv::Vec3b>(y1, x2), v3 = img2.at<cv::Vec3b>(y2, x1), v4 = img2.at<cv::Vec3b>(y2, x2);
                    blue += float(v1[0] + v2[0] + v3[0] + v4[0])/4;
                    green += float(v1[1] + v2[1] + v3[1] + v4[1])/4;
                    red += float(v1[2] + v2[2] + v3[2] + v4[2])/4;
                    exImg.at<cv::Vec3b>(h, w) = cv::Vec3b(blue, green, red);
                }
            }
        }
    }
    return exImg;
}

void matchImgs(cv::Mat &img1, cv::Mat img2, int maxFeatures=500)
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
    cv::Mat Tr = RANSAC(keyPoint1, keyPoint2, matches, img1, img2);
    cv::Mat exImg = spliceImgs(img1, img2, Tr);
    img1 = exImg;
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));
    std::vector<std::string> files = {"/data1", "/data2", "/data3", "/data4"};
    for(int fileId = 0; fileId < files.size(); fileId++) {
        std::vector<cv::Mat> imgs;
        std::string path = "../data" + files[fileId];
        readImgs(imgs, path);
        cv::Mat totalImg = imgs[1];
        for(int imgId = 0; imgId < imgs.size(); imgId++) if(imgId != 1) {
            matchImgs(totalImg, imgs[imgId]);
        }
        cv::Mat normalizedImg;
        cv::normalize(totalImg, normalizedImg, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::imwrite("./"+files[fileId]+".png", normalizedImg);
        // cv::namedWindow("MyWindow", cv::WINDOW_NORMAL);
        // cv::resizeWindow("MyWindow", totalImg.cols, totalImg.rows);
        // cv::imshow("MyWindow", totalImg);
        // cv::waitKey(0);
    }
}