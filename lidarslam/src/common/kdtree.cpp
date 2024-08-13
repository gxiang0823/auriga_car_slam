#include "kdtree.h"
#include <vector>

bool KdTree::BuildTree(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
    
    if (cloud->empty()) {
        return false;
    }
    
    Clear();                       //清理上一次建数的指针及其对象。
    root_.reset(new KdTreeNode()); //每次建树时创建一个新的根节点。
    
    cloud_ = cloud;

    std::vector<int> idx(cloud->size());
    for (int i = 0; i < cloud->points.size(); ++i) {
        idx[i] = i;
    }
    Insert(idx, root_.get());
    return true;
}

void KdTree::Clear()
{
    for (const auto &np : nodes_) {
        if (np.second != root_.get() && np.second != nullptr) {
            delete np.second;
        }
    }
    nodes_.clear();
    size_ = 0;
    tree_node_id_ = 0;
}

void KdTree::Insert(const std::vector<int> &points, KdTreeNode *node)
{
    nodes_.insert({node->id_, node});
    if (points.empty()) {
        return;
    }

    if (points.size() == 1) {
        size_++;
        node->point_idx_ = points[0];
        return;
    }

    std::vector<int> left, right;

    if (!FindSplitAxisAndThresh(points, node->axis_index_, node->split_thresh_, left, right)) {
        size_++;
        node->point_idx_ = points[0];
        return;
    }

    const auto create_if_not_empty = [&node, this](KdTreeNode *&new_node, const std::vector<int> &index) {
        if (!index.empty()) {
            new_node = new KdTreeNode;
            new_node->id_ = ++tree_node_id_;
            Insert(index, new_node);
        }
    };

    create_if_not_empty(node->left_, left);
    create_if_not_empty(node->right_, right);
}

bool KdTree::GetClosestPoint(const pcl::PointXYZI &pt, std::vector<int> &closest_idx, int k)
{
    if(k > size_){
        ROS_INFO("cannot set k=%d larger than cloud size: %d",k,size_);
        return false;
    }
    k_ = k;
    std::priority_queue<NodeAndDistance> knn_result;
    Knn(pt.getVector3fMap(), root_.get(), knn_result);

    // 排序并返回结果
    closest_idx.resize(knn_result.size());
    for (int i = closest_idx.size() - 1; i >= 0; --i) {
        // 倒序插入
        closest_idx[i] = knn_result.top().node_->point_idx_;
        knn_result.pop();
        //ROS_INFO("get closestdis2 %f",knn_result.top().distance2_);
    }
    //ROS_INFO("get closestpoint %d\t%d\t%d\t%d\t%d",closest_idx[0],closest_idx[1],closest_idx[2],closest_idx[3],closest_idx[4]);
    
    return true;
}

bool KdTree::GetClosestPointMT(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, std::vector<std::pair<size_t, size_t>> &matches, int k) {
    
    matches.resize(cloud->size() * k);

    // 索引
    std::vector<int> index(cloud->size());
    for (int i = 0; i < cloud->points.size(); ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [this, &cloud, &matches, &k](int idx) {
        std::vector<int> closest_idx;
        GetClosestPoint(cloud->points[idx], closest_idx, k);
        for (int i = 0; i < k; ++i) {
            matches[idx * k + i].second = idx;
            if (i < closest_idx.size()) {
                matches[idx * k + i].first = closest_idx[i];
            } else {
                matches[idx * k + i].first = std::numeric_limits<size_t>::max();
            }
        }
    });

    return true;
}

bool KdTree::FindSplitAxisAndThresh(const std::vector<int> &point_idx, int &axis, float &th, std::vector<int> &left, std::vector<int> &right)
{
    Eigen::Vector3f var;
    Eigen::Vector3f mean;
    int len = point_idx.size();
    mean = std::accumulate(point_idx.begin(),point_idx.end(),Eigen::Vector3f::Zero().eval(),[this](const Eigen::Vector3f &acc, const int &idx){return acc+cloud_->points[idx].getVector3fMap();})/len;
    var = std::accumulate(point_idx.begin(),point_idx.end(),Eigen::Vector3f::Zero().eval(),[&mean,this](const Eigen::Vector3f &acc, const int &idx){return acc+(cloud_->points[idx].getVector3fMap()-mean).cwiseAbs2();})/(len-1);    
    int max_i, max_j;
    var.maxCoeff(&max_i, &max_j);
    axis = max_i;
    th = mean[axis];

    for (const auto &idx : point_idx) {
        if (cloud_->points[idx].getVector3fMap()[axis] < th) {
            left.emplace_back(idx);
        } 
        else {
            right.emplace_back(idx);
        }
    }

    // 边界情况检查：输入的points等于同一个值，上面的判定是>=号，所以都进了右侧
    // 这种情况不需要继续展开，直接将当前节点设为叶子就行
    if (point_idx.size() > 1 && (left.empty() || right.empty())) {
        return false;
    }

    return true;
    
}

void KdTree::Knn(const Eigen::Vector3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const
{
    if (node->IsLeaf()) {
        // 如果是叶子，检查叶子是否能插入
        ComputeDisForLeaf(pt, node, knn_result);
        return;
    }

    // 看pt落在左还是右，优先搜索pt所在的子树
    // 然后再看另一侧子树是否需要搜索
    KdTreeNode *this_side, *that_side;
    if (pt[node->axis_index_] < node->split_thresh_) {
        this_side = node->left_;
        that_side = node->right_;
    } else {
        this_side = node->right_;
        that_side = node->left_;
    }

    Knn(pt, this_side, knn_result);
    if (NeedExpand(pt, node, knn_result)) {  // 注意这里是跟自己比
        Knn(pt, that_side, knn_result);
    }
}

void KdTree::ComputeDisForLeaf(const Eigen::Vector3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const
{
    // 比较与结果队列的差异，如果优于最远距离，则插入
    float dis2 = Dis2(pt, cloud_->points[node->point_idx_].getVector3fMap());
    if (knn_result.size() < k_) {
        // results 不足k
        knn_result.emplace(node, dis2);
    } else {
        // results等于k，比较current与max_dis_iter之间的差异
        if (dis2 < knn_result.top().distance2_) {
            knn_result.emplace(node, dis2);
            knn_result.pop();
        }
    }
}

bool KdTree::NeedExpand(const Eigen::Vector3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const
{
    if (knn_result.size() < k_) {
        return true;
    }
    // 检测切面距离，看是否有比现在更小的
    float d = pt[node->axis_index_] - node->split_thresh_;
    if ((d * d) < knn_result.top().distance2_ * alpha_) {
        return true;
    } else {
        return false;
    }
}

void KdTree::PrintAll() {
    for (const auto &np : nodes_) {
        auto node = np.second;
        if (node->left_ == nullptr && node->right_ == nullptr) {
            std::cout << "leaf node: " << node->id_ << ", idx: " << node->point_idx_<<std::endl;
        } else {
            std::cout << "node: " << node->id_ << ", axis: " << node->axis_index_ << ", th: " << node->split_thresh_<<std::endl;
        }
    }
}