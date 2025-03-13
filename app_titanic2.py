import sys

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Descripción del Desafío de Kaggle")

st.write("El Desafío del Titanic en Kaggle es uno de los concursos más populares para principiantes en el campo del "
         "análisis de datos y el aprendizaje automático. El objetivo es construir un modelo predictivo que determine "
         "si un pasajero sobrevivió o no al hundimiento del Titanic en función de las características disponibles "
         "sobre los pasajeros.")

st.markdown("> **Nota**: Puedes encontrar el desafío en el siguiente enlace: [Titanic: Machine Learning from Disaster]"
            "(https://)")

logo_url = ("https://historia.nationalgeographic.com.es/medio/2023/06/20/the-steamship-titanic-rmg-bhc3667_00000000"
            "_9b5bd117_230620084335_1280x775.jpg")
st.image(logo_url, width=400)

st.markdown("### Objetivo del Ejercicio")

st.write("El objetivo de este ejercicio es realizar un análisis exploratorio de datos (EDA) para comprender mejor "
         "el conjunto de datos del Titanic.")

st.write("Esto implica:")

st.markdown("""
1.   **Carga y visualización de los datos:** Importar los datos y familiarizarse con su estructura. 
2.   **Resumen estadístico y visualización de datos faltantes:** Identificar datos faltantes y obtener un resumen 
estadístico básico.
3.   **Análisis de variables clave:** Explorar cómo variables específicas, como el sexo, la clase y la edad, afectan 
la tasa de supervivencia.
4.   **Visualización de patrones y relaciones:** Utilizar gráficos para identificar patrones y relaciones en los datos.)
""")

st.markdown("### **Hipótesis a corroborar**:")

st.markdown("""
- **Hipótesis 01**: Los pasajeros de primera clase tuvieron la mayor tasa de supervivencia.
- **Hipótesis 02**: Las mujeres y los niños tuvieron una mayor tasa de supervivencia.
- **Hipótesis 03**: Los pasajeros que pagaron una tarifa más alta tuvieron una mayor tasa de supervivencia.
""")

st.markdown("## Análisis de Datos")

# Cargar el DataFrame desde un archivo CSV
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

st.markdown("### Datos del Titanic")
st.write("A continuación se muestra el DataFrame cargado desde un archivo CSV:")
st.dataframe(df)

# Opcional: Puedes usar st.table si prefieres una tabla estática
st.write("Tabla estática:")
st.table(df.head(10))  # Mostrar solo las primeras 10 filas

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.write("Información del DataFrame usando `df.info()`:")
st.text(s)

st.markdown("Con el comando info, podemos apreciar que hay tres columnas que presentan datos nulos (`Age`, "
            "`Cabin` y `Embarked`), "
            "los cuales debemos analizar para determinar el tratamiento que se les dará a continuación.")

# Calcular y mostrar el número de valores nulos por columna
valores_nulos = df.isnull().sum()

# Cambiar el nombre de las columnas
valores_nulos = valores_nulos.reset_index()
valores_nulos.columns = ['Columns', 'Null']

st.write("Número de valores nulos por columna:")
st.dataframe(valores_nulos)

# Calcular el promedio de la columna Age
promedio_age = df["Age"].mean().round()

# Reemplazar los valores nulos en la columna Age por el promedio
df["Age"] = df["Age"].fillna(promedio_age)

# Mostrar el DataFrame modificado
st.write("DataFrame con valores nulos reemplazados por el promedio de Age:")
st.write(df)

# cambiar los valores nulos de las columnas Cabin y Embarked, por 'unknow'
columns_to_replace = ["Cabin", "Embarked"]
for col in columns_to_replace:
    df[col] = df[col].fillna("unknown")

st.table(df.head(10))

st.write("Verificar si quedan valores nulos")
# Calcular y mostrar el número de valores nulos por columna
valores_nulos = df.isnull().sum()

# Cambiar el nombre de las columnas
valores_nulos = valores_nulos.reset_index()
valores_nulos.columns = ['Columns', 'Null']

st.write("Número de valores nulos por columna:")
st.dataframe(valores_nulos)

st.markdown("Se ha realizado una limpieza de datos en la que los valores nulos en la columna de edad (`Age`) se "
            "reemplazaron por el promedio, "
            "y los valores nulos en las columnas de cabinas (`Cabin`) y embarque (`Embarked`) se sustituyeron por "
            "'unknown'.")

st.markdown("Ahora, verificaremos si existen **datos duplicados**.")

# Título de la aplicación
st.markdown("### Contador de Filas Duplicadas")

# Mostrar el DataFrame
st.write("DataFrame:")
st.dataframe(df)

# Calcular y mostrar el número de filas duplicadas
num_duplicados = df.duplicated().sum()
st.write(f"Número de filas duplicadas: {num_duplicados}")

st.write("Análisis Estadístico")
st.dataframe(df.describe())

# Título de la aplicación
st.markdown("### Número de Valores Únicos por Columna")

# Mostrar el DataFrame
st.write("DataFrame:")
st.dataframe(df)

# Calcular y mostrar el número de valores únicos por columna
valores_unicos = df.nunique()
valores_unicos = valores_unicos.reset_index()
valores_unicos.columns = ["Columns", "Unique"]
st.write("Número de valores únicos por columna:")
st.write(valores_unicos)

st.markdown("## Análisis Exploratorio de Datos")

st.markdown("Ya que se ha hecho una limpieza de datos, se procederá ha hacer un análisis para determinar cuales fueron "
            "los factores más "
            "determinantes para la sobrevivencia de los pasajeros del Titanic.")

# Ver cantidad de sobrevivientes
sobrevivientes = df.loc[df["Survived"]==1]

if st.button("Sobrevivientes"):
    suma_sobrevivientes = sobrevivientes["Survived"].sum()
    st.write(f"Número de sobrevivientes: {suma_sobrevivientes}")
    
st.markdown("### Análisis de Sobrevivientes por Sexo")
    
# porcentaje de sobrevivientes respecto del sexo de estos
# Agrupar por 'Sex' y calcular la media de 'Survived', luego ordenar
promedio_sobrevivientes_por_sexo = (
    df[["Sex", "Survived"]]
    .groupby(['Sex'], as_index=False)
    .mean()
    .sort_values(by="Survived", ascending=False)
)
st.write("Promedio de sobrevivientes por sexo:")
st.write(promedio_sobrevivientes_por_sexo)

st.markdown("### Análisis de Sobrevivientes por Sexo")

st.write("Número de pasajeros que vivieron y murieron respecto del sexo de estos")
df_sex = df.groupby(["Sex","Survived"]).size().reset_index(name="Count")
st.write(df_sex)

# Crear la figura y la gráfica de barras
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x="Sex", y="Count", hue="Survived", data=df_sex, palette="Set2")

# Añadir los números encima de las barras
for p in barplot.patches:
    width = p.get_width()
    if width > 0:
        barplot.annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="center",
                         xytext=(0, 9),
                         textcoords='offset points')

# Título y etiquetas
plt.title("Número de Sobrevivientes y No Sobrevivientes por Sexo")
plt.xlabel("Sexo")
plt.ylabel("Número de Pasajeros")

# Ajustar la leyenda
handles, labels = barplot.get_legend_handles_labels()
barplot.legend(handles=handles, labels=["No", "Sí"], title="Sobrevivió")

# Mostrar la gráfica en Streamlit
st.pyplot(plt)

# Cierra la figura de matplotlib para evitar que Streamlit lo muestre dos veces
plt.close()

st.markdown("El gráfico anterior ilustra claramente la influencia del sexo en la tasa de supervivencia de los "
            "pasajeros. En total, habían más hombres que mujeres entre los pasajeros. La tasa de supervivencia es "
            "considerablemente mayor en mujeres que en hombres. La diferencia en la tasa de supervivencia entre "
            "géneros podría deberse a las políticas de evacuación del Titanic, que priorizaron a mujeres y niños. "
            "En resumen, el gráfico resalta una clara disparidad en la supervivencia entre hombres y mujeres durante"
            " el desastre del Titanic, con una notable ventaja de supervivencia para las mujeres.")

#Definir la función
def tabla_grafico(df, col1, col2):
    """
    Agrupar dos columnas, cuenta los grupos y los deja en formato df
    """
    df_group = df.groupby([col1, col2]).size().reset_index(name="Count")
    return df_group

st.markdown("### Análisis de Sobrevivientes por Edad")

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
columna = "Age"
variable_objetivo = "Survived"

def calculate_percentage_vo(df, col, bins, vo):
    # Agregar una nueva columna al DataFrame con los rangos de la columna específica
    df[col + 'Range'] = pd.cut(df[col], bins=bins, right=False)

    # Calcular el conteo de cada grupo y reestructurar los datos
    counts = df.groupby([vo, col + 'Range']).size().reset_index(name='Count')

    # Calcular los porcentajes por categoría
    counts['Percentage'] = counts.groupby(vo)['Count'].transform(lambda x: (x / x.sum()))

    # Crear una tabla pivote con los porcentajes
    pivot_counts = counts.pivot_table(values=['Count', 'Percentage'], index=col + 'Range', columns=vo)

    # Eliminar columna extra
    df.drop(col + 'Range', axis=1, inplace=True)

    return pivot_counts

# Crear la aplicación de Streamlit
st.markdown("Tabla de Edad por rangos y Sobrevivientes")

# Calcular la tabla de porcentajes
pivot_counts = calculate_percentage_vo(df, columna, bins, variable_objetivo)

# Mostrar la tabla de porcentajes
st.write("Tabla de Porcentajes")
st.dataframe(pivot_counts)

# Configurar el estilo de Seaborn
sns.set(style='whitegrid')

# Función principal
def main():
    st.markdown("#### Porcentaje de Sobrevivientes por Rango de Edad")

    # Definir los bins para los rangos de edad
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    # Agregar una nueva columna al DataFrame con los rangos de la columna específica
    df["AgeRange"] = pd.cut(df["Age"], bins=bins, right=False)

    # Calcular el conteo de cada grupo y reestructurar los datos
    counts = df.groupby(["Survived", "AgeRange"]).size().reset_index(name='Count')

    # Calcular los porcentajes por categoría
    counts["Percentage"] = counts.groupby("Survived")["Count"].transform(lambda x: (x / x.sum()))

    # Gráfico de barras con Seaborn
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=counts, x="AgeRange", y="Percentage", hue="Survived", palette="Set2")

    # Rotar los ejes x (45 grados) y añadir los valores en cada barra (excluyendo 0%)
    for p in ax.patches:
        if p.get_height() != 0:  # Si el valor no es 0%
            ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()), ha="center",
                        va="center", xytext=(0, 10), textcoords="offset points", fontsize=6)

    # Establecer título y etiquetas de los ejes
    plt.title("Porcentaje de Sobrevivientes por Rango de Edad")
    plt.xlabel("Rango de Edad")
    plt.ylabel("Porcentaje de Sobrevivientes")
    plt.xticks(rotation=0)  # Rotar etiquetas del eje x para mejor legibilidad
    plt.tight_layout()
    # Mostrar el gráfico
    st.pyplot(plt)
    # Eliminar columna extra del DataFrame
    df.drop("AgeRange", axis=1, inplace=True)

if __name__ == "__main__":
    main()

st.markdown("Podemos observar en el gráfico el porcentaje de sobrevivientes del Titanic por rangos de edad. "
            "La mayor tasa de supervivencia se encuentra en el rango de edad de [0-10] años. A medida que aumenta "
            "la edad, la probabilidad de supervivencia disminuye paulatinamente. El rango de edad de [30-40] años es "
            "el que contaba con más pasajeros y muestra los porcentajes más altos tanto de no sobrevivientes como "
            "de sobrevivientes, con una ligera mayoría de no sobrevivientes. En los adultos, la probabilidad de "
            "supervivencia es más equilibrada, mientras que los ancianos presentan una mayor tasa de mortalidad."
            "En conclusión, hubo una mayor probabilidad de supervivencia en los más jóvenes, seguida de una tasa"
            " más equilibrada entre los adultos y una notable disminución en los ancianos. Esta distribución puede"
            " reflejar factores como las prioridades en el rescate (por ejemplo, 'mujeres y niños primero') "
            "y la capacidad física para enfrentar la situación de emergencia.")

st.markdown("### Análisis de Sobrevivientes por Tarifas")

# Crear rango de tarifas
df["FareBand"] = pd.qcut(df["Fare"], 4)

# Formatear la columna FareBand para eliminar Left y Right
df["FareBand"] = df["FareBand"].apply(lambda x: f"{x.left:.2f} - {x.right:.2f}")

# Crear la tabla de resumen
summary_table = df[["FareBand", "Survived"]].groupby(["FareBand"], as_index=False).mean().sort_values(by="FareBand", ascending=True)

# Configuración de la aplicación Streamlit
st.markdown("Tabla de Rangos de Tarifa y Sobrevivientes")

st.write("Esta es la tabla que muestra el promedio de sobrevivientes en función de los rangos de tarifa:")

st.dataframe(summary_table)

# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
graph = sns.barplot(x="FareBand", y="Survived", data=summary_table, color="#fdb462", edgecolor="#fdb462")

# Configurar etiquetas y título
graph.set_xlabel("Rango de Tarifa")
graph.set_ylabel("Sobrevivientes")
graph.set_title("Sobrevivientes por Tarifa")

# Mostrar el gráfico en Streamlit
st.pyplot(plt)

st.markdown("Con los datos proporcionados, podemos concluir que la probabilidad de supervivencia era mayor para"
            " aquellos que habían pagado una tarifa más elevada.")

st.markdown("### Análisis de Sobrevivientes por Clases")

# Llamar a la función
st.write(tabla_grafico(df,"Pclass","Survived"))

# Definir la nueva función
def tabla_grafico_1(df, col, col_ob):
    df_col1 = df[[col, col_ob]].groupby([col], as_index=False).mean().sort_values(by=col_ob, ascending=False)
    return df_col1

# Llamar a la función
df_col1 = tabla_grafico_1(df, "Pclass", "Survived")
st.write(df_col1)

# Crear el gráfico de barras
plt = sns.barplot(x="Pclass", y="Survived", data=df_col1, color="#fdb462", edgecolor="#fdb462")

# Configurar etiquetas y título
plt.xlabel("Clases")
plt.ylabel("Sobrevivientes")
plt.title("Sobrevivientes por Clase")

# Mostrar el gráfico en Streamlit
st.pyplot(plt)

st.markdown("El gráfico muestra claramente que la clase fue un factor determinante en la supervivencia de los"
            " pasajeros. Los pasajeros de primera clase tuvieron la tasa de supervivencia más alta, seguidos por los"
            " de segunda clase, mientras que los de tercera clase tuvieron la menor tasa de supervivencia.")

st.markdown("### Análisis de Sobrevivientes por Familias")

# Llamar a la función
df_col1 = tabla_grafico_1(df, "SibSp", "Survived")
st.write(df_col1)

# Crear el gráfico de barras
sns.barplot(x="SibSp", y="Survived", data=df_col1, color="#fdb462", edgecolor="#fdb462")

# Configurar etiquetas y título
plt.xlabel("SibSp")
plt.ylabel("Sobrevivientes")
plt.title("Sobrevivientes por SibSp")

# Mostrar el gráfico en Streamlit
st.pyplot(plt)












