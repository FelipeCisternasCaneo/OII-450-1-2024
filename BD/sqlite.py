import sqlite3
import os
from Problem.SCP.problem import obtenerOptimo
from Problem.USCP.problem import obtenerOptimoUSCP
from Problem.USCP.problem import obtenerOptimoUSCP
from util import util

class BD:
    def __init__(self):
        self.__dataBase = 'D:/Descargas/OII-450-1-2024/BD/resultados.db'
        self.__conexion = None
        self.__cursor   = None

    def getDataBase(self):
        return self.__dataBase
    
    
    def setDataBase(self, dataBase):
        self.__dataBase = dataBase
        
        
    def getConexion(self):
        return self.__conexion
    
    
    def setConexion(self, conexion):
        self.__conexion = conexion
        
        
    def getCursor(self):
        return self.__cursor
    
    
    def setCursor(self, cursor):
        self.__cursor = cursor

    def conectar(self):
        conn = sqlite3.connect(self.getDataBase())
        cursor = conn.cursor()
        
        self.setConexion(conn)
        self.setCursor(cursor)
    
    def desconectar(self):
        self.getConexion().close()
        
    def commit(self):
        self.getConexion().commit()
        
    def construirTablas(self):
        self.conectar()
        
        self.getCursor().execute(
            ''' CREATE TABLE IF NOT EXISTS instancias(
                id_instancia INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo_problema TEXT,
                nombre TEXT,
                optimo REAL,
                param TEXT
            )'''
        )
        
        self.getCursor().execute(
            ''' CREATE TABLE IF NOT EXISTS experimentos(
                id_experimento INTEGER PRIMARY KEY AUTOINCREMENT,
                experimento TEXT,
                MH TEXT,
                paramMH TEXT,
                ML TEXT,
                paramML TEXT,
                ML_FS TEXT,
                paramML_FS TEXT,
                estado TEXT,
                fk_id_instancia INTEGER,
                FOREIGN KEY (fk_id_instancia) REFERENCES instancias (id_instancia)
            )'''
        )

        self.getCursor().execute(
            ''' CREATE TABLE IF NOT EXISTS resultados(
                id_resultado INTEGER PRIMARY KEY AUTOINCREMENT,
                fitness REAL,
                tiempoEjecucion REAL,
                solucion TEXT,
                fk_id_experimento INTEGER,
                FOREIGN KEY (fk_id_experimento) REFERENCES experimentos (id_experimento)
            )'''
        )

        self.getCursor().execute(
            ''' CREATE TABLE IF NOT EXISTS iteraciones(
                id_archivo INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT,
                archivo BLOB,
                fk_id_experimento INTEGER,
                FOREIGN KEY (fk_id_experimento) REFERENCES experimentos (id_experimento)
            )'''
        )
        
        self.commit()
        
        self.insertarInstanciasBEN()
        self.insertarInstanciasSCP()
        self.insertarInstanciasUSCP()

        self.desconectar()
    
    def insertarExperimentos(self, data, corridas, id):
        self.conectar()

        for corrida in range(corridas):
            self.getCursor().execute(f'''
                INSERT INTO experimentos VALUES (
                    NULL,
                    '{str(data["experimento"])}',
                    '{str(data["MH"])}',
                    '{str(data["paramMH"])}',
                    '{str(data["ML"])}',
                    '{str(data["paramML"])}',
                    '{str(data["ML_FS"])}',
                    '{str(data["paramML_FS"])}',
                    '{str(data["estado"])}',
                    {id}
                )''')
        
        
        self.commit()
        self.desconectar()
        
    
    def insertarInstanciasBEN(self):
        self.conectar()
        
        data = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23']
        
        data = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23']
        
        for instancia in data:
            tipoProblema = 'BEN'
            
            
            if instancia == 'F1':
                param     = f'lb:-100,ub:100'
                optimo = 0
                
                
            if instancia == 'F2':
                param     = f'lb:-10,ub:10'
                optimo = 0
                
                
            if instancia == 'F3':
                param     = f'lb:-100,ub:100'
                optimo = 0
                
                
            if instancia == 'F4':
                param     = f'lb:-100,ub:100'
                optimo = 0
                
                
            if instancia == 'F5':
                param     = f'lb:-30,ub:30'
                optimo = 0
                
                
            if instancia == 'F6':
                param     = f'lb:-100,ub:100'
                optimo = 0
                
                
            if instancia == 'F7':
                param     = f'lb:-1.28,ub:1.28'
                optimo = 0
                
                
            if instancia == 'F8':
                param     = f'lb:-500,ub:500'
                optimo = -418.9829
                
                
            if instancia == 'F9':
                param     = f'lb:-5.12,ub:5.12'
                optimo = 0
                
                
            if instancia == 'F10':
                param     = f'lb:-32,ub:32'
                optimo = 0
                
                
            if instancia == 'F11':
                param     = f'lb:-600,ub:600'
                optimo = 0
                
                
            if instancia == "F12":
                param     = f'lb:-50,ub:50'
                optimo = 0
                
                
            if instancia == "F13":
                param     = f'lb:-50,ub:50'
                optimo = 0
                
                
            if instancia == "F14":
                param     = f'lb:-65.536,ub:65.536'
                optimo = 1
                
                
            if instancia == "F15":
                param     = f'lb:-5,ub:5'
                optimo = 0.00030
                
                
            if instancia == "F16":
                param     = f'lb:-5,ub:5'
                optimo = -1.0316
                
                
            if instancia == "F17":
                param     = f'lb:-5,ub:5'
                optimo = 0.398
                
                
            if instancia == "F18":
                param     = f'lb:-2,ub:2'
                optimo = 3
                
                
            if instancia == "F19":
                param     = f'lb:0,ub:1'
                optimo = -3.86
                
                
            if instancia == "F20":
                param     = f'lb:0,ub:1'
                optimo = -3.32
                
                
            if instancia == "F21":
                param     = f'lb:0,ub:10'
                optimo = -10.1532
                
                
            if instancia == "F22":
                param     = f'lb:0,ub:10'
                optimo = -10.4028
                
                
            if instancia == "F23":
                param     = f'lb:0,ub:10'
                optimo = -10.5363
           
           
            self.getCursor().execute(f'''  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) ''', (tipoProblema, instancia, optimo, param))
            
        self.commit()
        self.desconectar()
        
    def insertarInstanciasSCP(self):
        self.conectar()
        
        data = os.listdir('D:/Descargas/OII-450-1-2024/Problem/SCP/Instances/')
        
        for d in data:
            tipoProblema = 'SCP'
            nombre = d.split(".")[0]
            optimo = obtenerOptimo(nombre)
            nombre = f'{nombre[3:]}'
            param = ''
            
            self.getCursor().execute(f'''  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) ''', (tipoProblema, nombre, optimo, param))
            
        self.commit()
        self.desconectar()
        
    def insertarInstanciasUSCP(self):
        self.conectar()
        
        data = os.listdir('D:/Descargas/OII-450-1-2024/Problem/USCP/Instances/')        
        for d in data:
            
            tipoProblema = 'USCP'
            nombre = d.split(".")[0]
            optimo = obtenerOptimoUSCP(nombre)
            
            nombre = f'u{nombre[4:]}'
            
            param = ''
            
            self.getCursor().execute(f'''  INSERT INTO instancias (tipo_problema, nombre, optimo, param) VALUES(?, ?, ?, ?) ''', (tipoProblema, nombre, optimo, param))
            
        self.commit()
        self.desconectar()
    
    def obtenerExperimento(self):
        conn = sqlite3.connect(self.getDataBase())
        conn.execute('BEGIN EXCLUSIVE')
        cursor = conn.cursor()
        cursor.execute(''' SELECT * FROM experimentos WHERE estado = 'pendiente' LIMIT 1''')
        data = cursor.fetchall()
        
        if data:
            experimento_id = data[0][0]
            cursor.execute(f''' UPDATE experimentos SET estado = 'ejecutando' WHERE id_experimento =  {experimento_id} ''')
            conn.commit()
            conn.close()
            
            
            return data
        
        
        else:
            conn.commit()
            conn.close()
            
            
            return None
    
    def obtenerExperimentos(self):
        self.conectar()
        
        cursor = self.getCursor()
        
        cursor.execute(''' SELECT * FROM experimentos WHERE estado = 'pendiente' ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        return data
    
    def obtenerInstancia(self,id):
        self.conectar()
        
        cursor = self.getCursor()
        
        cursor.execute(f''' SELECT * FROM instancias WHERE id_instancia = {id} ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        return data
    
    def actualizarExperimento(self, id, estado):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f''' UPDATE experimentos SET estado = '{estado}' WHERE id_experimento =  {id} ''')
        
        
        self.commit()
        self.desconectar()
        
    def insertarIteraciones(self, nombre_archivo, binary, id):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''  INSERT INTO iteraciones (nombre, archivo, fk_id_experimento) VALUES(?, ?, ?) ''', (nombre_archivo, binary, id))
        
        self.commit()
        self.desconectar()
        
    def insertarResultados(self, BestFitness, tiempoEjecucion, Best, id):
        self.conectar()
        
        cursor = self.getCursor()
        
        cursor.execute(f''' INSERT INTO resultados VALUES (
            NULL,
            {BestFitness},
            {round(tiempoEjecucion,3)},
            '{str(Best.tolist())}',
            {id}
        )''')
        
        self.commit()
        self.desconectar()
        
    def obtenerArchivos(self, instancia):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f''' 
            select i.nombre, i.archivo 
            from experimentos e 
            inner join iteraciones i on e.id_experimento  = i.fk_id_experimento 
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' 
            order by i2.nombre desc , e.MH desc   
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresArchivos(self, instancia, ml):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, i2.nombre  , i.nombre , i.archivo , MIN(r.fitness)  
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}'
            group by e.MH , i2.nombre
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresArchivosconClasificador(self, instancia, ml, ml_fs):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, e.ML_FS, i2.nombre  , i.nombre , i.archivo , MIN(r.fitness) 
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.ML_FS = '{ml_fs}'
            group by e.MH , i2.nombre  
            group by e.MH , i2.nombre  
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresArchivosconClasificadorBSS(self, instancia, ml, ml_fs,bss):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, e.ML_FS, e.paramMH, i2.nombre  , i.nombre , i.archivo , MIN(r.fitness), r.solucion   
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.paramMH = 'iter:100,pop:10,DS:V4-STD,cros:0.9;mut:0.20' and e.ML_FS = '{ml_fs}' and e.MH = '{bss}'
            group by e.MH , i2.nombre, e.paramMH  
            group by e.MH , i2.nombre, e.paramMH  
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresArchivosconBSS(self, instancia, ml, bss):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, e.ML_FS, i2.nombre  , i.nombre , i.archivo , MIN(r.fitness) 
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.paramMH like '%{bss}%' 
            group by e.MH , i2.nombre      
            group by e.MH , i2.nombre      
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresSoluciones(self, instancia, ml):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, r.solucion, MIN(r.fitness) 
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}'
            group by e.MH , i2.nombre   
            group by e.MH , i2.nombre   
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerArchivosBSSClasificador(self, instancia, ml, bss, clasificador):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, e.ML_FS, e.paramMH, i2.nombre  , i.nombre , i.archivo , r.fitness  
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.paramMH = 'iter:{bss},pop:40,DS:V4-STD,cros:0.9;mut:0.20' and e.ML_FS = '{clasificador}' and e.MH = 'GA'      
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.paramMH = 'iter:{bss},pop:40,DS:V4-STD,cros:0.9;mut:0.20' and e.ML_FS = '{clasificador}' and e.MH = 'GA'      
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerArchivosTecnica(self, instancia, ml, clasificador, tecnica):
        self.conectar()
        
        cursor = self.getCursor()
        cursor.execute(f'''             
            select e.id_experimento , e.MH , E.ML, e.ML_FS, e.paramMH, i2.nombre  , i.nombre , i.archivo , r.fitness  
            from resultados r 
            inner join experimentos e on r.fk_id_experimento = e.id_experimento
            inner join iteraciones i on i.fk_id_experimento = e.id_experimento
            inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.paramMH = 'iter:500,pop:50,DS:V4-STD,cros:0.9;mut:0.20' and e.ML_FS = '{clasificador}' and e.MH = '{tecnica}'        
            where i2.nombre  = '{instancia}' and e.ML = '{ml}' and e.paramMH = 'iter:500,pop:50,DS:V4-STD,cros:0.9;mut:0.20' and e.ML_FS = '{clasificador}' and e.MH = '{tecnica}'        
        ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerInstancias(self, problema):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' select DISTINCT id_instancia, nombre from instancias i where nombre in ({problema})   ''')
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerTecnicas(self):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' SELECT DISTINCT MH from experimentos e   ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerInstanciasEjecutadas(self, tipo_problema):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' select DISTINCT i.nombre  from experimentos e inner join instancias i on e.fk_id_instancia = i.id_instancia where i.tipo_problema = '{tipo_problema}' order by i.nombre asc ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    # def obtenerExperimentos(self, tipo_problema, mh):
    #     self.conectar()
    #     cursor = self.getCursor()
    #     cursor.execute(f''' SELECT DISTINCT e.experimento  from experimentos e inner join instancias i on e.fk_id_instancia = i.id_instancia where i.tipo_problema = '{tipo_problema}' AND e.MH = '{mh}' ''')
    #     data = cursor.fetchall()
        
        
    #     self.desconectar()
    #     return data
    
    def obtenerExperimentosEspecial(self, tipo_problema, mh, especial):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' SELECT DISTINCT e.experimento  from experimentos e inner join instancias i on e.fk_id_instancia = i.id_instancia where i.tipo_problema = '{tipo_problema}' AND e.MH = '{mh}' and e.experimento like '%{especial}%' ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerEjecuciones(self, instancia, mh, experimento):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f'''
        cursor.execute(f'''
                        select e.id_experimento , e.experimento, i.nombre , i.archivo , r.fitness, r.tiempoEjecucion  
                        from resultados r 
                        inner join experimentos e on r.fk_id_experimento = e.id_experimento
                        inner join iteraciones i on i.fk_id_experimento = e.id_experimento
                        inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
                        where i2.nombre  = '{instancia}' and e.experimento = '{experimento}' and e.MH = '{mh}'
                        ''')
        
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresEjecucionesSCP(self, instancia, mh, experimento):
        self.conectar()
        cursor = self.getCursor()
        cursor.execute(f''' 
    
    
                        select e.id_experimento , e.experimento, i.nombre , i.archivo , MIN(r.fitness) 
                        from resultados r 
                        inner join experimentos e on r.fk_id_experimento = e.id_experimento
                        inner join iteraciones i on i.fk_id_experimento = e.id_experimento
                        inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
                        where i2.nombre  = '{instancia}' and e.experimento = '{experimento}' and e.MH = '{mh}'
                        ''')
        
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerMejoresEjecucionesKP(self, instancia, mh, experimento):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' 
                       
                        select e.id_experimento , e.experimento, i.nombre , i.archivo , MAX(r.fitness) 
                        from resultados r 
                        inner join experimentos e on r.fk_id_experimento = e.id_experimento
                        inner join iteraciones i on i.fk_id_experimento = e.id_experimento
                        inner join instancias i2 on e.fk_id_instancia = i2.id_instancia 
                        where i2.nombre  = '{instancia}' and e.experimento = '{experimento}' and e.MH = '{mh}'
                        
                        ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerOptimoInstancia(self, instancia):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' SELECT optimo  from instancias i where nombre = '{instancia}'  ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    
    def obtenerTiposProblemas(self):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' 
            SELECT DISTINCT i.tipo_problema from experimentos e 
            inner join instancias i ON e.fk_id_instancia = i.id_instancia
                        ''')


        data = cursor.fetchall()
        
        
        return data
    
    def obtenerNombreExperimentos(self, tipo_problema):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' 
            SELECT DISTINCT experimento  
            from experimentos e
            inner join instancias i ON e.fk_id_instancia = i.id_instancia
            where i.tipo_problema = '{tipo_problema}'
                        ''')
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data
    
    def obtenerInstanciasByProblema(self, problema):
        self.conectar()
        
        
        cursor = self.getCursor()
        cursor.execute(f''' 
            SELECT DISTINCT i.nombre  from experimentos e 
            inner join instancias i ON e.fk_id_instancia = i.id_instancia
            where i.tipo_problema = '{problema}'
                        ''')
        
        
        data = cursor.fetchall()
        
        self.desconectar()
        
        
        return data