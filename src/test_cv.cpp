
#include <opencv2/opencv.hpp>

#include <string>

using namespace std;

int cv_test();
int canny_operator();
int canny_video();
int main(int argc, char** argv)
{
    // cv_test();
    // canny_operator();
    canny_video();
    return 0;
}
int cv_test()
{
    cv::Mat M1(3000,2000, CV_8UC1, cv::Scalar(100));
    M1.row(2000) = M1.row(1200)*2;
    cout << M1.row(20) << endl;
    uchar value1 = M1.at<uchar>(150,100);
    M1.at<uchar>(150,100)=0; //将第 150 行第 100 列像素值设置为 0
    cv::imshow("creat_img",M1);
    cv::waitKey(0);

    cv::Mat M3(300,200, CV_8UC3, cv::Scalar(0,0,255));
    // cout << M3 << endl;
    uchar value3 = M3.at<uchar>(150,100);
    M3.at<uchar>(150,100)=0; //将第 150 行第 100 列像素值设置为 0
    cv::imshow("creat_img3",M3);
    cv::waitKey(0);
    return 0;
}
int canny_operator()
{
    cv::Mat im = cv::imread("../index.jpeg", 0);
    if( im.empty() )
    {
        cout << "Can not load image." << endl;
        return -1;
    }
    cv::Mat result;
    cv::Canny(im, result, 50, 255);
    //保存结果
    cv::imwrite("kobe-canny.png", result);
    cv::imshow("index_img",result);
    cv::waitKey(0);
    return 0;
}
int canny_video()
{
    using namespace cv;
    //打开第一个摄像头
    //VideoCapture cap(0);
    //打开视频文件
    VideoCapture cap(0);
    //检查是否成功打开
    if(!cap.isOpened())
    {
        cerr << "Can not open a camera or file." << endl;
        return -1;
    }
    Mat edges;
    //创建窗口
    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        //从 cap 中读一帧,存到 frame
        cap >> frame;
        //如果未读到图像
        if(frame.empty())
            break;
        //将读到的图像转为灰度图
        cvtColor(frame, edges, CV_BGR2GRAY);
        //进行边缘提取操作
        Canny(edges, edges, 0, 30, 3);
        //显示结果
        imshow("edges", edges);
        //等待 30 秒,如果按键则推出循环
        if(waitKey(30) >= 0)
        break;
    }
    //退出时会自动释放 cap 中占用资源
    return 0;
}