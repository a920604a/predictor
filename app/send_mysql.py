import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from mysql_db import Detection, Model, Product, Site
from sqlite_db import make_model
from collections import namedtuple
from tqdm import tqdm

if __name__ == '__main__':
    det = make_model('t_detection')
    db_path = './detection.db'
    assert os.path.exists(db_path), 'check db file is exist'
    sqlite_engine = create_engine(
        'sqlite:///'+os.path.split(db_path)[1], echo=False)
    sqlite_session = sessionmaker(bind=sqlite_engine)

    mysql_engine = create_engine(
        'mysql+mysqldb://ayun:yuan@192.168.10.150:3306/devops?charset=utf8', echo=False)
    mysql_session = sessionmaker(bind=mysql_engine)

    mysql_sess = mysql_session()
    sqlite_sess = sqlite_session()

    # while True:
    ret = sqlite_sess.query(det).order_by('process_time').all()

    # start_data_process_time = ret[0].process_time.strftime(
    #     "%Y-%m-%d %H:%M:%S:%f")
    # if len(ret)>200:
    #     print('create new db')

    #     det2 = make_model('t_detection300')
    #     sqlite_engine2 = create_engine('sqlite:///detection3000.db', echo=True)
    #     det2.metadata.create_all(sqlite_engine2)
    #     sqlite_session2 = sessionmaker(bind=sqlite_engine2)

    #     start_data_process_time = ret[0].process_time.strftime(
    #         "%Y-%m-%d %H:%M:%S:%f")
    #     end_data_process_time = ret[-1].process_time.strftime(
    #         "%Y-%m-%d %H:%M:%S:%f")

    for res in (ret):
        res_pt = res.process_time.strftime("%Y-%m-%d %H:%M:%S:%f")
        print(res_pt)
    # break
    # model = mysql_sess.query(Model).filter_by(
    #     name=res.model_name).first()
    # product = mysql_sess.query(Product).filter_by(
    #     product_name=res.product_name).first()
    # site = mysql_sess.query(Site).filter_by(
    #     site_name=res.site_name).first()

    # detection = Detection()
    # detection.model_id = model.id,
    # detection.product_id = product.id,
    # detection.site_id = site.id,

    # detection.lot_number = res.lot_number,
    # detection.serial_number = res.serial_number,
    # detection.process_time = res.process_time,
    # detection.image_name = res.image_name,
    # detection.source_path = res.source_path,
    # detection.reference_path = res.reference_path,
    # detection.detection_path = res.detection_path,
    # detection.detection_class = res.detection_class,
    # detection.true_label = res.true_label,
    # detection.confidence = res.confidence

    # mysql_sess.add(detection)
    # mysql_sess.commit()

    # _ = sqlite_sess.query(det).filter_by(id=res.id).delete()
    # sqlite_sess.commit()

    sqlite_sess.close()
    mysql_sess.close()
