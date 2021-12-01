from pyspark.sql import functions as F


class EntitySet:
    def __init__(self):
        self.entities = {}
        self.relationship = {}

    def reset_df(self, name, df, index):
        assert name in self.entities
        self.entities[name] = (df, index)
        for relation in self.relationship[name]:
            if relation['on'] is not None and len(relation['on']) > 0:
                right = relation['right']
                print('on relation of {} and {} is not empty, may cause undefined behavior'.format(name, right))

    def add_df(self, name, df, index):
        assert name not in self.entities
        self.entities[name] = (df, index)
        self.relationship[name] = []

    def add_rs(self, left, right, left_on, right_on, time_cond=None, on=None):
        assert left in self.entities
        assert right in self.entities
        self.relationship[left].append(
            {
                'right': right,
                'left_on': left_on,
                'right_on': right_on,
                'time_cond': time_cond,
                'on': on
            }
        )

    def _get_on(self, left, relation):
        left_entity = self.entities[left][0]
        right_entity = self.entities[relation['right']][0]
        on = relation['on'] or []
        for left_on, right_on in zip(relation['left_on'], relation['right_on']):
            on.append(left_entity[left_on] == right_entity[right_on])

        for left_on, right_on, left_day, right_day in relation['time_cond']:
            if left_day is None:
                on.append(left_entity[left_on] < right_entity[right_on])
            else:
                on.append(F.datediff(left_entity[left_on], right_entity[right_on]) <= left_day)
            if right_day is None:
                on.append(left_entity[left_on] > right_entity[right_on])
            else:
                on.append(F.datediff(left_entity[left_on], right_entity[right_on]) >= -right_day)

        return on

    def _join(self, left_name):
        left_df, left_index = self.entities[left_name]
        output_df = left_df

        right_cols = []
        for relation in self.relationship[left_name]:
            # 递归join
            right_name = relation['right']
            right_df = self._join(right_name)
            on = self._get_on(left_name, relation)
            print("left='{}', right='{}', on={}".format(left_name, right_name, on))
            join_df = right_df.join(left_df, on=on, how='inner')

            # 转换timestamp
            right_index = self.entities[right_name][1]
            cols = [
                right_df[col].cast('string').alias(col) if dtype == 'timestamp' else right_df[col]
                for col, dtype in right_df.dtypes if col != right_index
            ]
            join_df = join_df.select(left_df[left_index], F.struct(cols).alias(right_name))

            # groupby打成array
            join_df = join_df.groupby(left_df[left_index]).agg(F.collect_list(right_name).alias(right_name))
            right_cols.append(join_df[right_name])

            # 与最终结果join
            output_df = output_df.join(join_df, on=left_index, how='left')

        # 打成struct
        if len(right_cols) > 0:
            output_df = output_df.select(
                *[left_df[col] for col in left_df.columns],
                F.struct(right_cols).alias(left_name)
            )
        return output_df

    def join(self, left_name):
        output_df = self._join(left_name)
        print(output_df)
        return output_df.withColumn(left_name, F.to_json(left_name))
