import os
import pickle
import json
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, Required, NumberRange
from wtforms import StringField, SubmitField, FileField, SelectField, IntegerField, DecimalField
from werkzeug.utils import secure_filename

import models

DATA_PATH = './../data'
app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'super_key'

Bootstrap(app)


def fit_model(type_model, params_model, path_to_train, target_name):
    params_model = json.loads(params_model)

    if type_model == "GB":
        model = models.GradientBoostingMSE(**params_model)
    else:
        model = models.RandomForestMSE(**params_model)

    data = pd.read_csv(path_to_train)

    try:
        target = data[target_name]
        data.drop(target_name, inplace=True, axis=1)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        app.logger.info("Вы нехороший человек!\
        Да, я не продумал защиту от невалидных данных ну и что!!")
        return None

    model.fit(data.values, target.values)
    app.logger.info("Модель закончила обучаться!")
    return model


class ChoiceModelForm(FlaskForm):
    choices = [("GB", "Gradient boosting"),
               ("RF", "Random Forest")]
    field = SelectField("Choose model", choices=choices, validators=[Required()])

    submit_next = SubmitField('Next')


# Как это в одну форму объединить???
class ChoiceParamsFormGB(FlaskForm):
    field_1 = IntegerField('n_estimators', validators=[NumberRange(min=1)], default=30)
    field_2 = IntegerField('max_depth', validators=[NumberRange(min=1)], default=4)
    field_3 = DecimalField('learing_rate', validators=[NumberRange(min=0.0001)], default=0.1)

    submit_back = SubmitField("Back")
    submit_next = SubmitField('Next')


class ChoiceParamsFormRF(FlaskForm):
    field_1 = IntegerField('n_estimators', validators=[NumberRange(min=1)], default=30)
    field_2 = IntegerField('max_depth', validators=[NumberRange(min=1)], default=100)

    submit_back = SubmitField("Back")
    submit_next = SubmitField('Next')


class ChoiceTrainDatasetForm(FlaskForm):
    file_path = FileField('Path to train data', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    field = StringField("Target name", validators=[DataRequired()])
    sumbit_back = SubmitField('Back')
    submit_next = SubmitField('Next')


class ChoiceTestDatasetForm(FlaskForm):
    file_path = FileField('Path to train data', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    field = StringField("Target name(для подсчета функции потерь)", validators=[DataRequired()])
    submit_next = SubmitField('Next')


@app.route('/settings/model', methods=['GET', 'POST'])
def choice_model():
    try:
        models_form = ChoiceModelForm()

        if models_form.validate_on_submit():
            app.logger.info(f"Choose:{models_form.field.data}")
            return redirect(url_for('choice_params', type_model=models_form.field.data))
        return render_template('from_form.html', form=models_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route("/settings/params", methods=['GET', 'POST'])
def choice_params():
    try:
        type_model = request.args.get("type_model")

        if type_model == "RF":
            params_form = ChoiceParamsFormRF()
        elif type_model == "GB":
            params_form = ChoiceParamsFormGB()
        else:
            app.logger.info('Exception: Неверный выбор модели')
            return redirect(url_for("choice_model"))

        if params_form.validate_on_submit():
            if params_form.submit_next.data:
                params = {
                    "n_estimators": int(params_form.field_1.data),
                    "max_depth": int(params_form.field_2.data),
                }
                if isinstance(params_form, ChoiceParamsFormGB):
                    params["learning_rate"] = float(params_form.field_3.data)

                app.logger.info(f"Choose:{params.items()}")
                json_params = json.dumps(params, indent=4)
                return redirect(url_for("choice_dataset",
                                        type_model=type_model, params_model=json_params))
            else:
                return redirect(url_for('choice_model'))
        return render_template('from_form.html', form=params_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route("/settings/dataset", methods=['GET', 'POST'])
def choice_dataset():
    try:
        type_model = request.args.get("type_model")
        params_model = request.args.get("params_model")

        file_form = ChoiceTrainDatasetForm()

        if file_form.validate_on_submit():
            if file_form.submit_next.data:
                f = file_form.file_path.data
                filename = secure_filename(f.filename)

                f.save(os.path.join(DATA_PATH, filename))
                app.logger.info(f"Save to data:{os.path.join(DATA_PATH, filename)}")
                return redirect(url_for("training_model",
                                        type_model=type_model,
                                        params_model=params_model,
                                        path_to_train=os.path.join(DATA_PATH, filename),
                                        target_name=file_form.field.data))
            else:
                return redirect(url_for('choice_params'))
        return render_template("from_form.html", form=file_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route("/training", methods=['GET', 'POST'])
def training_model():
    try:
        type_model = request.args.get("type_model")
        params_model = request.args.get("params_model")
        path = request.args.get("path_to_train")
        target_name = request.args.get("target_name")

        model = fit_model(type_model, params_model, path, target_name)

        pkl_filename = "pickle_model.pkl"
        with open(os.path.join(DATA_PATH, pkl_filename), 'wb') as file:
            pickle.dump(model, file)
        return redirect(url_for("get_predict"))
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for("choice_model"))


@app.route("/predict", methods=['GET', 'POST'])
def get_predict():
    try:
        file_form = ChoiceTestDatasetForm()

        if file_form.validate_on_submit():
            
            return redirect(url_for("training_model")
        return render_template("from_form.html", form=file_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
