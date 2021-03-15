from sqlalchemy import Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
import datetime
Base = declarative_base()


class Detection(Base):
    """
    预测记录
    """
    __tablename__ = 't_detection'

    id = Column(Integer, primary_key=True, autoincrement=True, comment="检测ID")
    model_id = Column(Integer, comment="模型ID")
    product_id = Column(Integer, comment="产品ID")
    site_id = Column(Integer, comment="站点ID")
    lot_number = Column(String(100), comment="批号")
    serial_number = Column(String(100), comment="序号")
    process_time = Column(TIMESTAMP(True), comment="检测时间",
                          nullable=False, server_default=datetime.datetime.now())
    image_name = Column(String(500), comment="图片名称")
    detection_path = Column(String(500), comment="检测图片路径")
    source_path = Column(String(500), comment="原图片路径")
    reference_path = Column(String(500), comment="标准图片路径")
    detection_class = Column(String(50), comment="检测类别")
    true_label = Column(String(50), comment="真实标签")
    description = Column(String(100), comment="描述")
    confidence = Column(Float, comment="置信度")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")

    create_by = Column(String(64), comment="创建者")
    created_at = Column(TIMESTAMP(True), comment="创建时间",
                        nullable=False, server_default=datetime.datetime.now())
    update_by = Column(String(64), comment="更新者")
    updated_at = Column(TIMESTAMP(True), comment="更新时间",
                        nullable=False, server_default=datetime.datetime.now())
    remark = Column(String(500), comment="备注")


class Dataset(Base):
    """
    数据集信息
    """
    __tablename__ = "t_dataset"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="数据集ID")
    name = Column(String(64), unique=True, comment="数据集名称")
    source_dataset_id = Column(Integer, comment="原始数据集")
    data_path = Column(String(1000), comment="数据集路径")
    times = Column(Integer, comment="倍数")
    import_method = Column(String(100), comment="导入方式")
    is_shift = Column(Integer, default=2, comment="是否平移（1是 2否）")
    is_flip = Column(Integer, default=2, comment="是否翻转（1是 2否）")
    is_random_point = Column(Integer, default=2, comment="是否随机点（1是 2否）")
    is_light = Column(Integer, default=2, comment="是否亮度（1是 2否）")
    is_noise = Column(Integer, default=2, comment="是否噪音（1是 2否）")
    is_warpaffine = Column(Integer, default=2, comment="是否形变（1是 2否）")
    is_labelme = Column(Integer, default=2, comment="是否已产生LabelMe（1是 2否）")
    is_coco = Column(Integer, default=2, comment="是否已产生COCO（1是 2否）")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")


class Defect(Base):
    """
    缺陷代码
    """
    __tablename__ = "t_defect"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="ID")
    dataset_id = Column(Integer, unique=True, comment="数据集ID")
    name = Column(String(64), unique=True, comment="缺陷代码")
    description = Column(String(64), comment="描述")
    confidence_threshold = Column(Float, comment="置信度阈值")
    order_num = Column(Integer, comment="序号")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")


class Model(Base):
    """
    模型信息
    """
    __tablename__ = "t_model"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="模型ID")
    name = Column(String(64), unique=True, comment="模型名称")
    algorithm = Column(String(100), comment="AI算法")
    image_width = Column(Integer, comment="图像宽度")
    image_height = Column(Integer, comment="图像高度")
    trainset_id = Column(Integer, unique=True, comment="训练集")
    testset_id = Column(Integer, unique=True, comment="测试集")
    validateset_id = Column(Integer, unique=True, comment="验证集")
    output_path = Column(String(100), comment="权重路径")
    dag_schedule = Column(String(100), comment="调度策略")
    epoch = Column(Integer, comment="训练轮次")
    batch_size = Column(Integer, comment="批次大小")
    learning_rate = Column(Float, comment="学习率")
    confidence_threshold = Column(Float, comment="置信度阈值")
    is_training = Column(Integer, default=2, comment="正在训练（1是 2否）")
    is_eval = Column(Integer, default=2, comment="正在评估（1是 2否）")
    is_pred = Column(Integer, default=2, comment="正在测试（1是 2否）")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")

    # def return_dict(self):
    #     return {
    #         "id": str(self.id),
    #         "name": self.name,
    #         "algorithm": self.algorithm,
    #         "image_width": str(self.image_width),
    #         "image_height": str(self.image_height),
    #         "trainset_id": str(self.trainset_id),
    #         "testset_id": str(self.testset_id),
    #         "validateset_id": str(self.validateset_id),
    #         "output_path": self.output_path,
    #         "dag_schedule": self.dag_schedule,
    #         "epoch": str(self.epoch),
    #         "batch_size":  str(self.batch_size) ,
    #         "learning_rate": str(self.learning_rate),
    #         "confidence_threshold": str(self.confidence_threshold),
    #         "is_training": str(self.is_training),
    #         "is_eval": str(self.is_eval),
    #         "is_pred": str(self.is_pred),
    #         "status": str(self.status)
    #     }


class Product(Base):
    """
    产品信息
    """
    __tablename__ = "t_product"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="ID")
    model_id = Column(Integer, unique=True, comment="模型ID")
    product_name = Column(String(64), unique=True, comment="产品名称")
    product_type = Column(String(64), unique=True, comment="产品类别")
    reference = Column(String(500), comment="引用信息")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")


class Site(Base):
    """
    站点信息
    """
    __tablename__ = "t_site"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="ID")
    site_name = Column(String(64), unique=True, comment="站点名称")
    site_type = Column(String(64), unique=True, comment="站点类别")
    reference = Column(String(500), comment="引用信息")
    status = Column(Integer, default=1, comment="状态（1正常 2停用）")
