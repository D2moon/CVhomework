#include <io.h>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

// #define TRAIN_MODEL 0   // 0表示测试，1表示训练
// #define TINY_IMAGE 1
// #define BAG_OF_SIFT 0

std::vector<std::string> list_directories(const std::string& path) {
    std::vector<std::string> list;
    std::string searchPath = path + "/*.*";

    intptr_t handle;
    _finddata_t fileInfo;

    handle = _findfirst(searchPath.c_str(), &fileInfo);

    if (handle == -1) {
        std::cerr << "Error opening directory: " << path << std::endl;
        return list;
    }

    do {
        if (fileInfo.attrib & _A_SUBDIR) {
            if (strcmp(fileInfo.name, ".") != 0 && strcmp(fileInfo.name, "..") != 0) {
                list.push_back(path + '/' + fileInfo.name);
            }
        }
    } while (_findnext(handle, &fileInfo) == 0);

    _findclose(handle);
    return list;
}

void read_images(std::vector<cv::Mat> &images, std::string path) {
    struct _finddata_t fileData;
    intptr_t findHandle = _findfirst((path + "/*.jpg").c_str(), &fileData);

    if (findHandle != -1) {
        do {
            cv::Mat image = cv::imread(path + "/" + fileData.name, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Failed to read image: " << fileData.name << std::endl;
                continue;
            }
            images.push_back(image);
        } while (_findnext(findHandle, &fileData) == 0);
        _findclose(findHandle);
    } else {
        std::cerr << "No matching files found in the directory" << std::endl;
    }
}

cv::Mat tiny_image(cv::Mat image, int size=16) {
    cv::Mat re;
    cv::resize(image, re, cv::Size(size, size));
    re = re.reshape(1, 1);
    re.convertTo(re, CV_32F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(re, mean, stddev);
    re = (re - mean.val[0]) / stddev.val[0];
    return re;
}

cv::Mat get_sift(std::vector<cv::Mat> images, int maxFeatures=100) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::Mat Features;
    sift->setNFeatures(maxFeatures);
    for(int i = 0; i < images.size(); i++) {
        std::vector<cv::KeyPoint> keyPoint;
        cv::Mat descriptor;
        sift->detectAndCompute(images[i], cv::Mat(), keyPoint, descriptor);
        Features.push_back(descriptor);
    }
    return Features;
}

cv::Ptr<cv::ml::KNearest> knn_model(cv::Mat train_images, cv::Mat train_labels, int k=5) {
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(k);
    knn->setIsClassifier(true);
    cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_images, cv::ml::ROW_SAMPLE, train_labels);
	knn->train(train_data);
	std::cout << "[Train] train knn done!" << std::endl;
    return knn;
}

int main() {
    std::string dir;
#if TRAIN_MODEL==1
    dir = "../train";
#else
    dir = "../test";
#endif
    cv::Mat DataMat, LabelMat;
    std::vector<std::string> list = list_directories(dir);
    std::vector<cv::Mat> images[list.size()];
    int rows = 0;
    for(int i = 0; i < list.size(); i++) {
        read_images(images[i], list[i]);
        rows += images[i].size();
    }

#if TINY_IMAGE==1
    int tiny_size = 16;
    DataMat = cv::Mat::zeros(rows, tiny_size * tiny_size, CV_32FC1);
    LabelMat = cv::Mat::zeros(rows, 1, CV_32SC1);
    int k = 0;
    for(int i = 0; i < list.size(); i++) {
        for(int j = 0; j < images[i].size(); j++) {
            cv::Mat image = tiny_image(images[i][j]);
            image.row(0).copyTo(DataMat.row(k++));
            LabelMat.at<int>(k, 0) = i;
        }
    }
#endif

#if BAG_OF_SIFT==1
    int k = 200, maxFeatures = 300;    // 聚类种类数
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->setNFeatures(maxFeatures);
    cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher);
    cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor = cv::makePtr<cv::BOWImgDescriptorExtractor>(sift, matcher);

    #if TRAIN_MODEL==1
    cv::Mat features;
    for(int i = 0; i < list.size(); i++) {
        cv::Mat f = get_sift(images[i], maxFeatures);
        features.push_back(f);
    }
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 300, 0.02);
    cv::BOWKMeansTrainer bowTrainer(k, criteria);
    bowTrainer.add(features);
    cv::Mat vocabulary = bowTrainer.cluster();
    bowExtractor->setVocabulary(vocabulary);

    cv::FileStorage fs("../models/bow_extractor_params.xml", cv::FileStorage::WRITE);
    fs << "vocabulary" << vocabulary;
    fs.release();
    #endif

    #if TRAIN_MODEL==0
    cv::FileStorage fs("../models/bow_extractor_params.xml", cv::FileStorage::READ);
    if (fs.isOpened()) {
        cv::Mat vocabulary;
        fs["vocabulary"] >> vocabulary;
        bowExtractor->setVocabulary(vocabulary);
        fs.release();
    } else {
        std::cerr << "Error: Unable to open the file 'bow_extractor_params.xml'" << std::endl;
        return -1;
    }
    #endif

    DataMat = cv::Mat::zeros(0, k, CV_32FC1);
    LabelMat = cv::Mat::zeros(0, 1, CV_32SC1);
    for(int i = 0; i < list.size(); i++) {
        for(int j = 0; j < images[i].size(); j++) {
            std::vector<cv::KeyPoint> keyPoint;
            cv::Mat descriptor;
            sift->detect(images[i][j], keyPoint);
            bowExtractor->compute(images[i][j], keyPoint, descriptor);
            DataMat.push_back(descriptor);
            LabelMat.push_back(i);
        }
    }
    std::cout << DataMat.rows << ' ' << DataMat.cols << std::endl;
    std::cout << LabelMat.rows << ' ' << LabelMat.cols << std::endl;
#endif

    std::string pos;
#if TINY_IMAGE==1
    pos = "../models/knn_model.xml";
#endif

#if BAG_OF_SIFT==1
    pos = "../models/sift_knn_model.xml";
#endif

#if TRAIN_MODEL==1
    cv::Ptr<cv::ml::KNearest> knn = knn_model(DataMat, LabelMat);
    knn->save(pos);
	std::cout << "[Train] save knn at " + pos << std::endl;
#else
    cv::Ptr<cv::ml::KNearest> knn = cv::Algorithm::load<cv::ml::KNearest>(pos);
    std::cout << "[Test] load knn at " + pos << std::endl;
    cv::Mat LabelHat;
    float ret = knn->predict(DataMat, LabelHat);
    LabelHat.convertTo(LabelHat, CV_32SC1);
    int equal_nums = 0;
	for (int i = 0; i <LabelHat.rows; i++)
	{
        // std::cout << LabelHat.at<int>(i, 0) << ' ' << LabelMat.at<int>(i, 0) << std::endl;
		if (LabelHat.at<int>(i, 0) == LabelMat.at<int>(i, 0))
		{
			equal_nums++;
		}
	}
	float acc = float(equal_nums) / float(LabelHat.rows);
    std::cout << "[Test] knn accuracy is " << acc << std::endl;
#endif
}