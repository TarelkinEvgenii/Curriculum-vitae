import os
import sys
from PyQt6 import QtWidgets, QtGui

from mainwindow import Ui_MainWindow

from pathlib import Path
from datetime import datetime
import ctypes
import ezdxf
import logging

import time
from PyQt6.QtCore import QRunnable  # контейнер для работы, которую вы хотите выполнить
from PyQt6.QtCore import QThreadPool  # Для многопоточности
from PyQt6.QtCore import pyqtSlot  # Для обозначения слотов
from PyQt6.QtWidgets import QApplication  # QApplication.processEvents() для получения ивентов в работающей функции
from PyQt6.QtWidgets import QMessageBox

from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout, QDialogButtonBox # для кастомного диалога для инструкций


from collections import Counter
import pandas as pd
import pyproj

from shapely.geometry import Polygon, LineString, LinearRing
import geopandas as gpd
import folium
import webbrowser


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        # Стандартная инициализация
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        # Дальнейшая инициализация
        # Инициализация обработчика
        self.processing_class = Processing()

        # Инициализация логгера
        now = datetime.now()
        self.now = f'{now.day:02}_{now.month:02}_{now.year:04} {now.hour:02}_{now.minute:02}_{now.second:02}'
        self.logger = self.processing_class.initialize_logger(self.now)
        # Инициализация многопоточности
        self.threadpool = QThreadPool()
        self.logger.info("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Доп. инициализация окна
        self.window().setFixedSize(430, 340) # Зафиксируем масштаб
        self.window().setWindowTitle(f'{self.window().windowTitle()} v1')
        # Настройка картинки окна
        basedir = os.path.dirname(__file__)
        self.setWindowIcon(QtGui.QIcon(os.path.join(basedir, 'red_lines_ico_newer.ico')))
        # Чтобы отображалась иконка корректно в панели задач
        try:
            from ctypes import windll  # Only exists on Windows.
            myappid = 'ru.kzn.irg.redlines.evgenii.first'
            windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except ImportError:
            pass


        # Инициализация кнопок
        # Окно поиска файла
        self.find_file.clicked.connect(self.find_file_dialog)
        # Окно загрузки dxf
        self.load_dxf_to_program.clicked.connect(self.load_dxf_dialog)
        # Переменная для отслеживания модификации dxf
        self.dxf_modified = False

        # Блокировка кнопки нахождения пересечений
        self.execute.setEnabled(False)
        self.execute.clicked.connect(self.red_line_crossing)

        # Настройка кнопок импорта
        self.save_to_xls.setEnabled(False)
        self.save_to_xls.clicked.connect(self.save_to_xlsx_function)

        self.save_to_html.setEnabled(False)
        self.save_to_html.clicked.connect(self.save_to_html_function)

        self.save_to_dxf.setEnabled(False)
        self.save_to_dxf.clicked.connect(self.save_to_dxf_function)

        # Настройка кнопки "Инструкция по слоям"
        self.instructions.clicked.connect(self.instructions_function)

    @pyqtSlot()
    def instructions_function(self):
        self.logger.info(f'Открытие инструкции по слоям')
        text = '''
Если в выпадающих списках, которые отвечают за выбор слоёв, вы увидели непонятные символы в названии слоёв и/или названия слоёв стали неразборчивы, то ваш конвертор в dxf не корректно конвертирует русский язык.

Самый простой вариант решения данный проблемы: открыть данный файл dxf в AutoCad, далее выбрать `Сохранить как` и выбрать один из последних форматов dxf (dxf 2015, dxf 2018 или более новые форматы). 

Всё проблема решена. 
AutoCad сам исправил некорректную конвертация русского языка.
        '''
        QMessageBox.information(self, 'Инструкция по слоям', text)
    @pyqtSlot()
    def find_file_dialog(self):
        file_filter = 'AutoCad dxf File (*.dxf)'
        initial_filter = 'AutoCad dxf File (*.dxf)'

        # Выбор пути
        if self.file_path.text() !='':
            if Path(self.file_path.text()).is_dir():
                path = self.file_path.text()
            else:
                path = os.getcwd()
        else:
            path = os.getcwd()

        response = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=path,
            filter=file_filter,
            initialFilter=initial_filter
        )
        # Если что-то выбрали
        if str(response[0])!='':
            self.file_path.setText(str(response[0]))

    @pyqtSlot()
    def load_dxf_dialog(self):

        self.logger.info(f"Start reading DXF file.")

        self.label_5.setText(f"Файл читается<br>Ожидайте")

        # Блокировка кнопки загрузки dxf файла
        self.load_dxf_to_program.setEnabled(False)
        # Блокировка кнопок импорта
        self.save_to_dxf.setEnabled(False)
        self.save_to_html.setEnabled(False)
        self.save_to_xls.setEnabled(False)

        self.label_6.setText('Eщё не искали')

        # Почистим CombotBox от предыдущих слоёв
        self.border.clear()
        self.areas.clear()
        self.red_lines.clear()

        QApplication.processEvents()  # Вызываем, чтобы применить обновление текста
        # (Функция выше даёт возможность приложению получить другие события,
        # но это не совсем хороший случай,
        # так как может получить долгое событие и будет тупить в ожидании его выполнения)

        worker = ThreadWorker(self.logger, self.load_dxf_multithreading, self.file_path.text())  # Any other args, kwargs are passed to the run function
        self.threadpool.start(worker)


    def load_dxf_multithreading(self, thread_file_path):

        file_read = False
        # файл ли?
        if os.path.isfile(self.file_path.text()):
            try:
                self.label_5.setText(f"Файл читается<br>Ожидайте")
                self.doc = ezdxf.readfile(self.file_path.text())
                file_read = True
                self.label_5.setText(f"Файл прочитан")
            except IOError:
                self.logger.info(f"Not a DXF file or a generic I/O error.")
                self.label_5.setText(f"Not a DXF file<br>or a generic I/O error.")
                # sys.exit(1)
            except ezdxf.DXFStructureError:
                self.logger.info(f"Invalid or corrupted DXF file.")
                self.label_5.setText(f"Invalid or corrupted DXF file.")
                # sys.exit(2)
        else:
            self.label_5.setText(f"Файл не выбран")


        # Проверка файл
        if file_read:
            self.logger.info(f"AutoCad release: {self.doc.acad_release}")

            self.msp = self.doc.modelspace()
            # Вывод количества аттрибутов посреди разных слоёв

            self.logger.info(f'There are following layers with following number of entities')

            self.layers = Counter(
                [e.dxf.layer for e in self.msp]).most_common()  # Различные слои и количество записей

            self.attributes_per_layer = {}  # Словарь с аттрибутами слоя

            self.different_dxftype = {}  # Словарь с типами данных в конкретных слоях

            for layer in self.layers:
                self.logger.info(layer)

            for layer in self.layers:
                selected_layer = layer[0]
                self.logger.info('*' * 60)
                self.logger.info(f'{selected_layer} layer has following DXFTYPE')
                self.different_dxftype[selected_layer] = Counter(
                    [e.DXFTYPE for e in self.msp if e.dxf.layer == selected_layer]).most_common()
                for dxf_type_objects in self.different_dxftype[selected_layer]:
                    self.logger.info(f'{dxf_type_objects}')

            for layer in self.layers:
                selected_layer = layer[0]
                self.logger.info('-*' * 60)
                # Проверка есть ли в слое INSERT тип
                flag_have_insert_type = max([element[0] == 'INSERT' for element in self.different_dxftype[selected_layer]])
                if flag_have_insert_type:
                    # Тут берём только INSERT объекты
                    self.logger.info(
                        f'Following layer selected: {selected_layer} -  and this layer has following number of attributes:')
                    try:
                        # Тут берём только INSERT объекты
                        number_of_attributes_per_layer = Counter(
                            [len(element.attribs) for element in
                             list([e for e in self.msp if e.dxf.layer == selected_layer and e.DXFTYPE == 'INSERT'])]
                        ).most_common()
                        for attribute in number_of_attributes_per_layer:
                            self.logger.info(f'\t{attribute}')
                        self.logger.info('\t' + '*' * 60)

                        # Вывод различных типов атрибутов, которые есть в слое
                        different_attributes = Counter(
                            [item.dxf.tag for sublist in
                             [element.attribs for element in
                              list([e for e in self.msp if e.dxf.layer == selected_layer and e.DXFTYPE == 'INSERT'])]
                             for item in sublist]
                        ).most_common()
                        for attribute in different_attributes:
                            self.logger.info(f'\t{attribute}')
                        # Сохранение различных аттрибутов слоя
                        self.attributes_per_layer[selected_layer] = different_attributes

                    except:
                        self.logger.info(f'ERROR! Something went wrong')
                        raise ValueError
                else:
                    self.logger.info(f'Following layer: {selected_layer} -  has no attributes:')

            self.label_5.setText(f"Завершено")

            # Заполняем выпадающие списки
            self.logger.info('Starting to fill in combo boxes')
            self.border.addItems([layer[0] for layer in self.layers])
            self.areas.addItems([layer[0] for layer in self.layers])
            self.red_lines.addItems([layer[0] for layer in self.layers])


            # Разблокировка кнопки нахождения пересечений
            self.execute.setEnabled(True)

            # Разблокировка кнопки загрузки файла
            self.load_dxf_to_program.setEnabled(True)

            # Когда файл загружен, то мы разрешаем модифицировать его
            self.dxf_modified = False
        else:
            self.load_dxf_to_program.setEnabled(True)
            self.label_5.setText(f"dxf файл не загружен")


    @pyqtSlot()
    def red_line_crossing(self):

        self.logger.info(f"Start of processing")

        self.label_6.setText(f"Идёт обработка<br>Ожидайте")
        QApplication.processEvents()  # Вызываем, чтобы применить обновление текста
        # (Функция выше даёт возможность приложению получить другие события,
        # но это не совсем хороший случай,
        # так как может получить долгое событие и будет тупить в ожидании его выполнения)
        worker = ThreadWorker(self.logger, self.red_line_crossing_multithreading)  # Any other args, kwargs are passed to the run function
        self.threadpool.start(worker)

    def red_line_crossing_multithreading(self):

        # Для начала считаем требуемые слои
        # self.border.currentText()
        # self.areas.currentText()
        # self.red_lines.currentText()
        self.logger.info('Start saving to csv')
        # Сохраним данные слоёв в файлы, чтобы в случае ошибок можно было посмотреть, что было там записано
        self.processing_class.getting_data_from_layer(self.msp, self.logger, self.attributes_per_layer, self.border.currentText(), self.now)
        self.processing_class.getting_data_from_layer(self.msp, self.logger, self.attributes_per_layer, self.areas.currentText(), self.now)
        self.processing_class.getting_data_from_layer(self.msp, self.logger, self.attributes_per_layer, self.red_lines.currentText(), self.now)
        self.logger.info('Finish saving to csv')
        # Считываем слои

        # Считываем данные
        column_with_coords = 'block_coords'
        column_with_coords_str = column_with_coords + '_str'
        resulting_column = 'geometry'

        self.logger.info('Start reading csv files')
        areas = pd.read_csv(f'.logs/{self.areas.currentText()}.csv')
        red_lines = pd.read_csv(f'.logs/{self.red_lines.currentText()}.csv')
        border = pd.read_csv(f'.logs/{self.border.currentText()}.csv')
        self.logger.info('Finish reading csv files')

        # Выделяем, то что хотим видеть в итоговой таблице
        self.summary_table_columns = list(areas.columns)
        # Удаляем первые совпадения
        self.summary_table_columns.remove('block_coords')
        self.summary_table_columns.remove('is_closed')
        self.summary_table_columns.remove('block_coords_len')

        # Делаем предобработку
        areas = self.processing_class.first_preprocessing(self.logger, areas)
        red_lines = self.processing_class.first_preprocessing(self.logger, red_lines)
        border = self.processing_class.first_preprocessing(self.logger, border)

        # Теперь превратим координаты в объект
        # для домов и земельных участков будем использовать Polygon, так как все объекты замкнуты
        # для дорог LineString, так как часть объектов замкнута, а часть не замкнута

        # areas GeoPandas
        areas[column_with_coords_str] = areas[column_with_coords].apply(lambda x: Polygon(x))

        # Превращаем в текст, чтобы распарсить именно при помощи GeoPandas
        areas[column_with_coords_str] = areas[column_with_coords_str].astype(str)
        areas[resulting_column] = gpd.GeoSeries.from_wkt(areas[column_with_coords_str])

        areas = gpd.GeoDataFrame(areas, geometry=resulting_column)

        # red_lines GeoPandas

        red_lines[column_with_coords_str] = red_lines[column_with_coords].apply(lambda x: LineString(x))

        # Превращаем в текст, чтобы распарсить именно при помощи GeoPandas
        red_lines[column_with_coords_str] = red_lines[column_with_coords_str].astype(str)
        red_lines[resulting_column] = gpd.GeoSeries.from_wkt(red_lines[column_with_coords_str])

        red_lines = gpd.GeoDataFrame(red_lines, geometry=resulting_column)

        # border GeoPandas

        border[column_with_coords_str] = border[column_with_coords].apply(lambda x: Polygon(x))

        # Превращаем в текст, чтобы распарсить именно при помощи GeoPandas
        border[column_with_coords_str] = border[column_with_coords_str].astype(str)
        border[resulting_column] = gpd.GeoSeries.from_wkt(border[column_with_coords_str])

        border = gpd.GeoDataFrame(border, geometry=resulting_column)

        # Частичная проверка, что хорошо сконвертировалось
        '''
        Проверка будет работать всегда, за исключением 
        Так как в areas могут быть полилинии у которой совпадает первая и последняя точка, 
        но эта полилиния не замкнута 
        и полилинию преобразуем в Poligon, а Polygon отображает только точки все без совпадения начала и конца,
        так как в случае Polygon - это делается не явно
        '''
        try:
            assert min(areas[resulting_column].apply(lambda x: len(list(x.exterior.coords))) == \
                       areas[column_with_coords].apply(lambda x: len(x)))
            assert min(red_lines[resulting_column].apply(lambda x: len(list(x.coords))) == \
                       red_lines[column_with_coords].apply(lambda x: len(x)))
        except Exception as inst:
            # logger.exception(inst)
            self.logger.exception(f'!!!!! Полилиния с одинаковым началом и концом, но не замкнута')

        # Поиск участков внутри границ
        border_inside = gpd.sjoin(border, areas,
                                  how='inner',
                                  predicate='intersects',
                                  lsuffix='border',
                                  rsuffix='areas')


        # Преобразуем из геометрии границы в геометрию участков
        column_with_coords_str = column_with_coords_str + '_areas'  # 'block_coords_str_areas'
        after_border_resulting_column = resulting_column + '_areas'  # 'geometry_areas'

        border_inside[after_border_resulting_column] = gpd.GeoSeries.from_wkt(border_inside[column_with_coords_str])
        border_inside = border_inside.set_geometry(after_border_resulting_column)

        border_inside = gpd.GeoDataFrame(border_inside,
                                         geometry=after_border_resulting_column)  # зачем дублировал делание геометрии

        # Пересекаем участки с красными линиями
        self.areas_intersection = gpd.sjoin(border_inside, red_lines,
                                       how='inner',
                                       predicate='intersects',
                                       lsuffix='border_inside',
                                       rsuffix='red_lines')

        # areas_intersection['block_coords'] - координаты красных линий
        # areas_intersection['block_coords_areas'] - координаты кадастровых участков

        # Кооридинаты MSK16
        self.msk_border = column_with_coords + '_border'
        self.msk_areas = column_with_coords + '_areas'

        # так как block_coords нет в border_inside, то будет название просто block_coords
        # А не block_coords_red_lines
        self.msk_roads = column_with_coords

        # Координаты WGS84
        self.wgs_areas = self.msk_areas + '_WGS'
        self.wgs_roads = self.msk_roads + '_WGS'

        # Разблокировка кнопок импорта
        self.save_to_dxf.setEnabled(True)
        self.save_to_html.setEnabled(True)
        self.save_to_xls.setEnabled(True)
        # Блокировка кнопки пересечений
        self.execute.setEnabled(False)

        self.label_6.setText(f"Завершено")
    @pyqtSlot()
    def save_to_html_function(self):
        self.logger.info(f"Start saving in html")

        self.label_7.setText(f"Идёт обработка<br>Ожидайте")
        QApplication.processEvents()  # Вызываем, чтобы применить обновление текста
        # (Функция выше даёт возможность приложению получить другие события,
        # но это не совсем хороший случай,
        # так как может получить долгое событие и будет тупить в ожидании его выполнения)

        # Выбор корректного пути сохранения
        if self.file_path.text() != '':
            path_of_folder_from_dxf_file = os.path.dirname(os.path.abspath(self.file_path.text()))
            if not Path(path_of_folder_from_dxf_file).is_dir():
                path_of_folder_from_dxf_file = os.getcwd()
        else:
            path_of_folder_from_dxf_file = os.getcwd()

        # Диалоги почему-то не работают в потоке
        file_filter = 'html file (*.html)'
        initial_filter = 'html file (*.html)'

        self.logger.info(f'Выбор файла для сохранения')

        response = QtWidgets.QFileDialog.getSaveFileName(parent=None,
                                                         caption='Select a data file',
                                                         directory=f'{path_of_folder_from_dxf_file}\\result_{self.now}.html',
                                                         filter=file_filter,
                                                         initialFilter=initial_filter)

        if str(response[0]) != '':
            self.logger.info(f'Начинаю сохранение')
            worker = ThreadWorker(self.logger,
                                  self.save_to_html_function_multithreading,
                                  response[0])  # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker)
        else:
            self.logger.info(f'Без сохранения, так как файл не был выбран')
            button = QMessageBox.warning(self, "Warning", "Не был выбран путь сохранения")

            if button == QMessageBox.StandardButton.Ok:
                self.logger.info("OK!")


    def save_to_html_function_multithreading(self, path):

        # Визуализация
        m = folium.Map(location=[55.794759, 49.102046], zoom_start=12)

        # areas_intersection
        areas_layer = folium.FeatureGroup(name='Земельные участки')

        # Участки
        self.logger.info("Вырисовка участков")

        areas_intersection_to_save = self.areas_intersection

        areas_intersection_to_save[self.wgs_areas] = areas_intersection_to_save[
            self.msk_areas].apply(lambda x: self.processing_class.convert_msk16_to_wgs84(x))

        # tooltip='London'
        necessarily_1 = 'Кадастровый номер, обозначение, учетный номер объекта'
        necessarily_2 = 'Кадастровый номер квартала'
        necessarily_3 = 'Вид использования по документу'
        necessarily_4 = 'Площадь, кв.м'
        for indx in range(len(areas_intersection_to_save)):
            try:
                folium.Polygon(locations=areas_intersection_to_save[self.wgs_areas].iloc[indx],
                               color='green',
                               fill=True,
                               fill_color='green',
                               popup=f'<b>{necessarily_1}</b>: <br>{areas_intersection_to_save[necessarily_1].iloc[indx]}<br><br>'
                                     f'<b>{necessarily_2}</b>: <br>{areas_intersection_to_save[necessarily_2].iloc[indx]}<br><br>'
                                     f'<b>{necessarily_3}</b>: <br>{areas_intersection_to_save[necessarily_3].iloc[indx]}<br><br>'
                                     f'<b>{necessarily_4}</b>: <br>{areas_intersection_to_save[necessarily_4].iloc[indx]}<br><br>').add_to(
                    areas_layer)
            except:
                folium.Polygon(locations=areas_intersection_to_save[self.wgs_areas].iloc[indx],
                               color='green',
                               fill=True,
                               fill_color='green').add_to(
                    areas_layer)
                self.logger.info('Ошибка заранее предустановленные значения столбцов для слоя участки отличаются от значений в текущих данных')
        # areas_intersection[self.msk_areas].apply(
        #     lambda x: folium.Polygon(locations=x, color='green', fill=True, fill_color='green', popup='<b>Area</b>').add_to(areas_layer))

        areas_layer.add_to(m)

        # Дороги
        self.logger.info("Вырисовка дорог")
        roads_layer = folium.FeatureGroup(name='Дороги')
        areas_intersection_to_save[self.wgs_roads] = areas_intersection_to_save[
            self.msk_roads].apply(lambda x: self.processing_class.convert_msk16_to_wgs84(x))

        areas_intersection_to_save[self.wgs_roads].apply(
            lambda x: folium.PolyLine(locations=x, color='red').add_to(roads_layer))

        roads_layer.add_to(m)
        # Граница
        self.logger.info("Вырисовка границы")
        border_layer = folium.FeatureGroup(name='Граница')
        border_for_visualization = self.processing_class.convert_msk16_to_wgs84(areas_intersection_to_save[
                                                              self.msk_border].iloc[
                                                              0])
        folium.PolyLine(locations=border_for_visualization, color='black').add_to(border_layer)

        border_layer.add_to(m)

        # Визуализация
        folium.LayerControl().add_to(m)

        m.save(f'{path}')

        self.logger.info(f'Finish saving in html {path} - {self.now}')

        webbrowser.open(f'{path}')

        self.label_7.setText(f"Завершено")


    @pyqtSlot()
    def save_to_xlsx_function(self):
        self.logger.info(f"Start saving in xlsx")

        self.label_7.setText(f"Идёт обработка<br>Ожидайте")
        QApplication.processEvents()  # Вызываем, чтобы применить обновление текста
        # (Функция выше даёт возможность приложению получить другие события,
        # но это не совсем хороший случай,
        # так как может получить долгое событие и будет тупить в ожидании его выполнения)

        # Выбор корректного пути сохранения
        if self.file_path.text() != '':
            path_of_folder_from_dxf_file = os.path.dirname(os.path.abspath(self.file_path.text()))
            if not Path(path_of_folder_from_dxf_file).is_dir():
                path_of_folder_from_dxf_file = os.getcwd()
        else:
            path_of_folder_from_dxf_file = os.getcwd()

        # Диалоги почему-то не работают в потоке
        file_filter = 'Excel File (*.xlsx)'
        initial_filter = 'Excel File (*.xlsx)'

        self.logger.info(f'Выбор файла для сохранения')

        response = QtWidgets.QFileDialog.getSaveFileName(parent=None,
                                                         caption='Select a data file',
                                                         directory=f'{path_of_folder_from_dxf_file}\\result_{self.now}.xlsx',
                                                         filter=file_filter,
                                                         initialFilter=initial_filter)



        if str(response[0]) != '':
            self.logger.info(f'Начинаю сохранение')
            worker = ThreadWorker(self.logger,
                                  self.save_to_xlsx_function_multithreading,
                                  response[0])  # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker)
        else:
            self.logger.info(f'Без сохранения, так как файл не был выбран')
            button = QMessageBox.warning(self, "Warning", "Не был выбран путь сохранения")

            if button == QMessageBox.StandardButton.Ok:
                self.logger.info("OK!")


    def save_to_xlsx_function_multithreading(self, path):

        if self.msk_roads not in self.summary_table_columns:
            self.summary_table_columns.append(self.msk_roads)
        if self.msk_areas not in self.summary_table_columns:
            self.summary_table_columns.append(self.msk_areas)

        areas_intersection_to_save = self.areas_intersection[self.summary_table_columns]

        areas_intersection_to_save[self.msk_roads] = areas_intersection_to_save[
            self.msk_roads].astype(str)
        areas_intersection_to_save[self.msk_areas] = areas_intersection_to_save[
            self.msk_areas].astype(str)

        areas_intersection_to_save.drop_duplicates(inplace=True)

        areas_intersection_to_save.reset_index(inplace=True, drop=True)

        # areas_intersection['block_coords'] - column_red_lines_with_coords_intersection - координаты красных линий
        # areas_intersection['block_coords_areas'] - column_areas_with_coords_intersection - координаты кадастровых участков

        # Чтобы не менять в оригинальном файле названия столбцов
        areas_intersection_to_save = areas_intersection_to_save.rename(
                                            columns={self.msk_roads: 'Координаты_красных_линий',
                                                     self.msk_areas: 'Координаты_кадастровых_участков'})

        areas_intersection_to_save.to_excel(f'{path}', index=False)
        self.logger.info(f'Finish saving in xlsx {path} - {self.now}')

        self.label_7.setText(f"Завершено")

    @pyqtSlot()
    def save_to_dxf_function(self):
        self.logger.info(f"Start saving in dxf")

        self.label_7.setText(f"Идёт обработка<br>Ожидайте")
        QApplication.processEvents()  # Вызываем, чтобы применить обновление текста
        # (Функция выше даёт возможность приложению получить другие события,
        # но это не совсем хороший случай,
        # так как может получить долгое событие и будет тупить в ожидании его выполнения)

        # Выбор корректного пути сохранения
        if self.file_path.text() != '':
            path_of_folder_from_dxf_file = os.path.dirname(os.path.abspath(self.file_path.text()))
            if not Path(path_of_folder_from_dxf_file).is_dir():
                path_of_folder_from_dxf_file = os.getcwd()
        else:
            path_of_folder_from_dxf_file = os.getcwd()

        # Диалоги почему-то не работают в потоке
        file_filter = 'AutoCad File (*.dxf)'
        initial_filter = 'AutoCad File (*.dxf)'

        self.logger.info(f'Выбор файла для сохранения')

        response = QtWidgets.QFileDialog.getSaveFileName(parent=None,
                                                         caption='Select a data file',
                                                         directory=f'{path_of_folder_from_dxf_file}\\result_{self.now}.dxf',
                                                         filter=file_filter,
                                                         initialFilter=initial_filter)

        if str(response[0]) != '':
            self.logger.info(f'Начинаю сохранение')
            worker = ThreadWorker(self.logger,
                                  self.save_to_dxf_function_multithreading,
                                  response[0])  # Any other args, kwargs are passed to the run function
            self.threadpool.start(worker)
        else:
            self.logger.info(f'Без сохранения, так как файл не был выбран')
            button = QMessageBox.warning(self, "Warning", "Не был выбран путь сохранения")

            if button == QMessageBox.StandardButton.Ok:
                self.logger.info("OK!")

    def save_to_dxf_function_multithreading(self, path):
        # После инициализации dxf ставим какую то переменную в True,
        # а если новый файл загружаем по False ставим
        if self.dxf_modified == False:
            # Добавляем слой
            self.logger.info(f'Добавляем слой в dxf')
            self.doc.layers.add(name="Mistakes", color=ezdxf.colors.RED)

            # Отрисовка зон
            self.logger.info(f'Начало отрисовки зон')
            areas = pd.DataFrame(self.areas_intersection[self.msk_areas])
            areas = areas.astype(str)
            # Удаление дубликатов
            areas.drop_duplicates(inplace=True)
            areas.reset_index(inplace=True, drop=True)
            areas = pd.DataFrame(areas[self.msk_areas].apply(lambda x: eval(x)))

            areas[self.msk_areas].apply(lambda x: self.msp.add_lwpolyline(x,
                                                                          dxfattribs={'layer': 'Mistakes',
                                                                                      'color': ezdxf.colors.BLUE}))

            # Отрисовка дорог
            self.logger.info(f'Начало отрисовки дорог')
            roads = pd.DataFrame(self.areas_intersection[self.msk_roads])
            roads = roads.astype(str)
            # Удаление дубликатов
            roads.drop_duplicates(inplace=True)
            roads.reset_index(inplace=True, drop=True)
            roads = pd.DataFrame(roads[self.msk_roads].apply(lambda x: eval(x)))

            roads[self.msk_roads].apply(lambda x: self.msp.add_lwpolyline(x,
                                                                          dxfattribs={'layer': 'Mistakes',
                                                                                      'color': ezdxf.colors.YELLOW}))

            # Изменили dxf
            self.dxf_modified = True
        # Сохраняем файл
        if self.dxf_modified == True:
            self.doc.saveas(f"{path}")
            self.logger.info(f'Finish saving in dxf {path} - {self.now}')
            self.label_7.setText(f"Завершено")
        else:
            self.logger.info(f'ERROR saving in dxf {path} - {self.now}')
            self.label_7.setText(f"Ошибка сохранения")
class ThreadWorker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, logger, fn, *args):
        super(ThreadWorker, self).__init__()
        # Initialize logger
        self.logger = logger
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.logger.info(f'Передано в функцию для нового потока: {str(*self.args)}')
        self.fn(*self.args)


class Processing:

    def initialize_logger(self, now: str):
        # Инициализация логгера
        log_folder = '.logs'

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            # Делаем папку для логов скрытой
            FILE_ATTRIBUTE_HIDDEN = 0x02
            ret = ctypes.windll.kernel32.SetFileAttributesW(log_folder, FILE_ATTRIBUTE_HIDDEN)

        log_format = '[%(asctime)s] %(name)-25s %(levelname)-8s %(message)s'
        logging.basicConfig(
            format=log_format,
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(str(log_folder) + f"/debug {now}.log", mode='a')
            ]
        )
        logger = logging.getLogger(__name__)
        return logger

    def getting_data_from_layer(self, msp, logger, attributes_per_layer, following_layer, now):

        logger.info(f'Inside {following_layer} layer inside object INSERT')
        inside_block = Counter(
            [item.DXFTYPE for e in msp if e.dxf.layer == following_layer and e.DXFTYPE == 'INSERT' for item in
             e.block()]).most_common()
        if len(inside_block) == 0:
            logger.info('\tEMPTY')
        else:
            for block_atrb in inside_block:
                logger.info(f'\t{block_atrb}')

        # Наличие в слое атрибутов
        try:
            columns_for_data_area = [attribute[0] for attribute in attributes_per_layer[following_layer]]
        except:
            columns_for_data_area = []

        columns_for_data_area.append('block_coords')  # координаты полилинии
        columns_for_data_area.append('is_closed')
        columns_for_data_area.append('DXFTYPE')
        columns_for_data_area.append('ERROR')

        data_csv = pd.DataFrame(columns=columns_for_data_area,
                                index=range(len([e for e in msp if e.dxf.layer == following_layer])))

        index_for_iteration = 0

        for e in msp:
            if e.dxf.layer == following_layer:
                block_coords = []
                dxftype = []
                is_closed = []
                if e.dxf.dxftype == 'INSERT':
                    # достаём данные из блока
                    for element in e.block():
                        # Тут обрабатывает любое количество полилиний
                        if element.DXFTYPE == 'LWPOLYLINE':
                            block_coords.insert(-1, list(
                                element.lwpoints.values))  # Не на последнее место ставит, а на предпоследнее
                            is_closed.insert(-1, element.is_closed)  # сделать is_closed списком

                        # Если внутри блок есть
                        elif element.DXFTYPE == 'INSERT':
                            logger.info(f'\tERROR BLOCK INSIDE BLOCK\t' * 3)
                            block_coords = []
                            data_csv.iloc[index_for_iteration]['ERROR'] = True
                            for inside in element.block():
                                if inside.DXFTYPE == 'LWPOLYLINE':
                                    block_coords.insert(-1, list(
                                        inside.lwpoints.values))  # Не на последнее место ставит, а на предпоследнее
                                    is_closed.insert(-1, inside.is_closed)

                    # достаём характеристики объекта
                    for attrib in e.attribs:
                        tag = attrib.dxf.tag
                        text = attrib.dxf.text
                        data_csv.iloc[index_for_iteration][tag] = text  # сохраняем значение тега
                    data_csv.iloc[index_for_iteration]['block_coords'] = block_coords
                    data_csv.iloc[index_for_iteration]['is_closed'] = is_closed
                    index_for_iteration += 1

                # Обработка полилиний
                elif e.dxf.dxftype == 'LWPOLYLINE':
                    block_coords.insert(-1, list(e.lwpoints.values))  # Не на последнее место ставит, а на предпоследнее
                    is_closed.insert(-1, e.is_closed)
                    dxftype.insert(-1, e.dxf.dxftype)

                    data_csv.iloc[index_for_iteration]['block_coords'] = block_coords
                    data_csv.iloc[index_for_iteration]['is_closed'] = is_closed
                    data_csv.iloc[index_for_iteration]['DXFTYPE'] = dxftype
                    index_for_iteration += 1
                elif e.dxf.dxftype == 'LINE':
                    start = list(e.dxf.start)
                    start.append(float(0))
                    start.append(float(0))

                    end = list(e.dxf.end)
                    end.append(float(0))
                    end.append(float(0))
                    coords = start + end

                    block_coords.insert(-1, list(coords))
                    is_closed.insert(-1, False)
                    dxftype.insert(-1, e.dxf.dxftype)

                    data_csv.iloc[index_for_iteration]['block_coords'] = block_coords
                    data_csv.iloc[index_for_iteration]['is_closed'] = is_closed
                    data_csv.iloc[index_for_iteration]['DXFTYPE'] = dxftype
                    index_for_iteration += 1

        data_csv.dropna(how='all', inplace=True)  # to drop if all values IN THE ROW are nan
        data_csv['block_coords_len'] = data_csv['block_coords'].apply(
            lambda x: len(x))  # количество объектов, которые будем отрисовать

        data_csv.to_csv(f'.logs/{following_layer}.csv', index=False)

        error_data_csv = data_csv.loc[data_csv.ERROR == True]
        if len(error_data_csv) != 0:
            error_data_csv.to_csv(f'ERROR_{following_layer} {now}.csv', index=False, encoding='UTF16')

    def msk16_to_wgs84(self, x, y):
        # Define the MSK16 and WGS84 coordinate reference systems
        msk16_crs = pyproj.CRS(
            "+proj=tmerc +lat_0=0 +lon_0=49.03333333333 +k=1 +x_0=1300000 +y_0=-5709414.70 +ellps=krass "
            "+towgs84=23.57,-140.95,-79.8,0,0.35,0.79,-0.22 +units=m +no_defs")
        wgs84_crs = pyproj.CRS('EPSG:4326')

        # Create a transformer to convert from MSK16 to WGS84
        msk16_to_wgs84_transformer = pyproj.Transformer.from_crs(msk16_crs, wgs84_crs)

        # Use the transformer to convert the input x and y coordinates to latitude and longitude
        latlng = msk16_to_wgs84_transformer.transform(x, y)

        return latlng

    def check_column_values(self, df: pd.DataFrame, col_name: str) -> bool:
        """
        Проверяет 3-5 коориднату в листе листов на наличие значения отличного от 0
        :param df:
        :param col_name:
        :return:
        """
        column = df[col_name].apply(lambda x: eval(x))
        for row_idx, row in enumerate(column):
            for my_list in row:
                # Iterate over each set of 5 elements in the row (which is a list)
                for i in range(0, len(my_list), 5):
                    # Iterate over every 3rd, 4th, and 5th element in the current set of 5 elements
                    for j in range(i + 2, i + 5):
                        if my_list[j] != 0:
                            return False

        return True

    def convert_msk16_to_wgs84(self, coords):
        wgs_coords = []
        for coord in coords:
            latlng = self.msk16_to_wgs84(coord[0], coord[1])
            wgs_coords.append(list(latlng))
        return wgs_coords

    def redundancy_reduction(self, list_for_reduction: list) -> list:
        resulting_list = []
        try:
            assert len(list_for_reduction) % 5 == 0
        except Exception as inst:
            self.logger.exception(inst)
            raise

        for i in range(int(len(list_for_reduction) / 5)):
            point = [list_for_reduction[i * 5], list_for_reduction[i * 5 + 1]]  # [x , y]
            resulting_list.append(point)
        return resulting_list

    def first_preprocessing(self,
                            logger,
                            data: pd.DataFrame,
                            column_with_coords: str = 'block_coords',
                            column_with_is_closed: str = 'is_closed') -> pd.DataFrame:
        '''
        Функция преобразует координаты в удобный формат после исполнения getting_data_from_dxf.py

        ЯВНО УЧИТЫВАЯ колонку column_with_is_closed, которая показывает замкнут или нет объект
        ЕСТЬ ОБЪЕКТ ЗАМКНУТ, то явно добавляем в конец строки первую точку объекта

        :param data:
        :param column_with_coords:
        :param column_with_is_closed:
        :return:
        '''
        self.logger = logger
        # 1. Проверяем есть ли где-то нули помимо x и y
        try:
            assert self.check_column_values(data, column_with_coords)
        except Exception as inst:
            self.logger.exception(inst)
            raise
        # 2. Превращаем из строк в списки
        data[column_with_coords] = data[column_with_coords].apply(lambda x: eval(x))
        data[column_with_is_closed] = data[column_with_is_closed].apply(lambda x: eval(x))
        # 3. Делаем explode
        # Для проверки корректности будущей строка ниже
        checking = data[column_with_coords].apply(lambda x: len(x)).value_counts().reset_index()
        checking_2 = data[column_with_is_closed].apply(lambda x: len(x)).value_counts().reset_index()
        assert len(data.explode(column_with_coords, ignore_index=True)) == len(
            data.explode(column_with_is_closed, ignore_index=True))

        data = data.explode([column_with_coords, column_with_is_closed],
                            ignore_index=True)  # параметр ignore_index важен,
        # чтобы в индексах не было дубликатов
        try:
            assert sum(checking['count'] * checking[column_with_coords]) == len(data)
            assert sum(checking_2['count'] * checking_2[column_with_is_closed]) == len(data)
            assert sum(checking['count'] * checking[column_with_coords]) == \
                   sum(checking_2['count'] * checking_2[column_with_is_closed])
        except Exception as inst:
            self.logger.exception(inst)
            raise
        # !!!!!! below for python 3.8 above for python 3.10!!!!
        # assert sum(checking['index'] * checking[column_with_coords]) == len(data)
        # 4. Убираем лишние нули и превращаем в удобный для библиотеки список
        data[column_with_coords] = data[column_with_coords].apply(
            lambda x: self.redundancy_reduction(x))
        # 5. Чтобы явно закрыть контур, если он закрыт
        # Чтобы не возникали двусмысленности
        '''
        У Polygon и LineRing
        Последовательность может быть явно закрыта путем передачи одинаковых значений в первый и последний индексы. 
        В противном случае последовательность будет НЕЯВНО закрыта копированием первого кортежа в последний индекс. 
        '''
        for indx in range(len(data)):
            # Проверка закрыт или нет, а потом проверка равен ли первый индекс последнему
            if data[column_with_is_closed].iloc[indx] == True and \
                    data[column_with_coords].iloc[indx][0] != data[column_with_coords].iloc[indx][-1]:
                data[column_with_coords].iloc[indx].append(
                    data[column_with_coords].iloc[indx][0])

        return data





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    # app.exec() # Не помогло