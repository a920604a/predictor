from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime

from sqlalchemy import Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base


def make_model(table_name: str):
    Base = declarative_base()

    class Detection(Base):
        """
        预测记录
        """
        __tablename__ = table_name
        id = Column(Integer, primary_key=True,
                    autoincrement=True, comment="检测ID")
        model_name = Column(String(100), comment="模型")
        product_name = Column(String(100), comment="产品")
        site_name = Column(String(100),  comment="站点")
        lot_number = Column(String(100), comment="批号")
        serial_number = Column(String(100), comment="序号")
        process_time = Column(TIMESTAMP(True), comment="检测时间",
                              nullable=False)
        image_name = Column(String(500), comment="图片名称")
        detection_path = Column(String(500), comment="检测图片路径")
        source_path = Column(String(500), comment="原图片路径")
        reference_path = Column(String(500), comment="标准图片路径")
        detection_class = Column(String(50), comment="检测类别")
        true_label = Column(String(50), comment="真实标签")
        # description = Column(String(100), comment="描述")
        confidence = Column(Float, comment="置信度")
        # status = Column(Integer, default=1, comment="状态（1正常 2停用）")

        create_by = Column(String(64), comment="创建者")
    return Detection
    
